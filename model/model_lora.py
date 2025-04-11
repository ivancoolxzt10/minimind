import torch
from torch import optim, nn

# 定义Lora网络结构
class LoRA(nn.Module):
    """
    定义 LoRA 适配器模块本身。
    它由两个线性层 A 和 B 组成，模拟低秩分解 W_delta = B * A。
    """
    def __init__(self, in_features, out_features, rank):
        """
        初始化 LoRA 模块。
        Args:
            in_features (int): 原始线性层的输入特征维度。
            out_features (int): 原始线性层的输出特征维度。
            rank (int): LoRA 的秩，控制中间低秩矩阵的维度，是一个关键超参数。
                        rank 越小，引入的参数越少，但可能表达能力受限；rank 越大，参数越多，可能表达能力更强。
        """
        super().__init__()
        self.rank = rank  # 存储秩的大小
        # 矩阵 A：将输入从 in_features 映射到低秩空间 rank。bias=False 表示不需要偏置项。
        self.A = nn.Linear(in_features, rank, bias=False)
        # 矩阵 B：将低秩空间 rank 映射回输出空间 out_features。bias=False。
        self.B = nn.Linear(rank, out_features, bias=False)
        # 初始化权重：这是 LoRA 的推荐初始化方式
        # 矩阵 A 使用小的正态分布初始化，有助于训练开始时的探索。
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵 B 初始化为全零。这意味着在训练开始时，LoRA 模块的输出是 0，
        # 模型的行为完全等同于原始模型，保证了训练的稳定性。
        # 随着训练进行，B 的权重会逐渐更新，LoRA 的效果才会显现。
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        LoRA 模块的前向传播。
        计算 W_delta * x = B * A * x。
        Args:
            x (torch.Tensor): 输入张量。
        Returns:
            torch.Tensor: LoRA 适配器的输出，即低秩更新量。
        """
        # 先通过 A 降维，再通过 B 升维。
        return self.B(self.A(x))

def apply_lora(model, rank=16):
    """
    将 LoRA 适配器应用到给定的模型中。
    它会遍历模型的所有模块，找到符合条件的线性层，并为其添加 LoRA 模块。
    Args:
        model (nn.Module): 需要应用 LoRA 的 PyTorch 模型。
        rank (int, optional): LoRA 的秩。默认为 16。
    """
    for name, module in model.named_modules(): # 遍历模型的所有命名模块 (包括子模块)
        # 检查当前模块是否是线性层 (nn.Linear)
        # 并且是一个方阵 (输入维度等于输出维度)。这个条件可能过于严格，
        # 实际应用中通常会选择性地应用于 Attention 中的 Q, K, V 投影层，
        # 即使它们不一定是方阵。这里的条件可能需要根据具体模型调整。
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 1. 创建 LoRA 模块实例
            # 输入输出维度与当前线性层一致，指定 rank。
            # .to(model.device) 确保 LoRA 模块和模型在同一个设备上 (CPU 或 GPU)。
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)

            # 2. 将 LoRA 模块添加到原始线性层模块上
            # 使用 setattr 将 lora 实例作为原始 nn.Linear 模块的一个新属性，名字叫 "lora"。
            # 这样可以通过 module.lora 访问到它。
            setattr(module, "lora", lora)

            # 3. 修改原始线性层的前向传播方法 (forward)
            # 保存原始的 forward 方法
            original_forward = module.forward

            # 定义一个新的 forward 方法，它结合了原始层和 LoRA 层的计算。
            # 使用显式绑定 (layer1=original_forward, layer2=lora) 来捕获当前的 original_forward 和 lora 实例，
            # 防止闭包问题导致引用错误。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 计算原始线性层的输出: original_output = W * x
                original_output = layer1(x)
                # 计算 LoRA 适配器的输出: lora_output = B * A * x
                lora_output = layer2(x)
                # 将两者相加: W * x + B * A * x
                return original_output + lora_output

            # 用新的 forward_with_lora 替换掉原始模块的 forward 方法。
            # 现在，当调用这个线性层时，它会自动执行包含 LoRA 的计算。
            module.forward = forward_with_lora

def load_lora(model, path):
    """
    从文件中加载预训练好的 LoRA 权重到模型中。
    Args:
        model (nn.Module): 已经应用了 LoRA 的模型 (即运行过 apply_lora)。
        path (str): LoRA 权重文件的路径 (.pt 或 .pth 文件)。
    """
    # 加载 LoRA 权重文件，map_location=model.device 确保加载到模型所在的设备。
    state_dict = torch.load(path, map_location=model.device)
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查模块是否拥有 "lora" 属性 (意味着 apply_lora 时已添加 LoRA 模块)
        if hasattr(module, 'lora'):
            # 从加载的 state_dict 中提取属于当前模块 LoRA 子模块的权重。
            # LoRA 权重在文件中通常是以 "父模块名.lora.A.weight" 或 "父模块名.lora.B.weight" 的形式存储的。
            # 这段代码构建一个只包含当前 LoRA 模块权重的字典 lora_state。
            # k.replace(f'{name}.lora.', '') 去掉了文件中的前缀，得到 LoRA 模块内部的参数名 (如 "A.weight")。
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 使用 load_state_dict 将提取到的权重加载到当前模块的 lora 子模块中。
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    """
    保存模型中所有 LoRA 适配器的权重到文件。
    只保存 LoRA 模块 (A 和 B 矩阵) 的权重，不保存原始模型权重。
    Args:
        model (nn.Module): 训练好的、包含 LoRA 模块的模型。
        path (str): 保存 LoRA 权重的目标文件路径。
    """
    state_dict = {} # 初始化一个空字典来存储所有 LoRA 权重
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查模块是否拥有 "lora" 属性
        if hasattr(module, 'lora'):
            # 获取当前 LoRA 子模块的状态字典 (包含 "A.weight", "B.weight")
            lora_module_state = module.lora.state_dict()
            # 为 LoRA 状态字典中的每个键添加前缀 "父模块名.lora."，以便区分不同层的 LoRA 权重。
            # 例如，"A.weight" -> "layers.0.attention.wq.lora.A.weight"
            lora_state = {f'{name}.lora.{k}': v for k, v in lora_module_state.items()}
            # 将当前 LoRA 模块的权重更新到总的 state_dict 中。
            state_dict.update(lora_state)
    # 使用 torch.save 将包含所有 LoRA 权重的字典保存到指定路径。
    torch.save(state_dict, path)