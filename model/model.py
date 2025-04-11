import math
import struct
import inspect
import time

# 假设 LMConfig 在本地文件 '.LMConfig' 中定义
from .LMConfig import LMConfig
# typing 用于类型注解，提高代码可读性和健壮性
from typing import Any, Optional, Tuple, List, Union
# numpy 用于数值计算，虽然这里主要用 torch，但可能在某些辅助环节用到
import numpy as np
# torch 是主要的深度学习框架
import torch
# F 包含 torch 的函数式接口，如激活函数、损失函数等
import torch.nn.functional as F
# nn 包含构建神经网络层的模块
from torch import nn
# 从 Hugging Face Transformers 库导入基类，方便集成和使用其生态
from transformers import PreTrainedModel
# Hugging Face 标准的模型输出结构，包含隐藏状态、logits、过去的 K/V 缓存等
from transformers.modeling_outputs import CausalLMOutputWithPast


# RMSNorm 层：一种比 LayerNorm 更简单的归一化层，LLaMA 等模型常用
class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (均方根层归一化).
    原理：通过输入的均方根来归一化激活值，然后乘以一个可学习的权重。
    优点：计算上比 LayerNorm 简单，有时效果相当或更好。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm 层。
        Args:
            dim (int): 输入张量的维度 (通常是 embedding 维度)。
            eps (float): 加在分母上的小值，防止除以零，增加数值稳定性。
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """ 计算 RMS 归一化核心部分 """
        # 计算均方根: sqrt(mean(x^2))
        # x.pow(2): 计算 x 的平方
        # .mean(-1, keepdim=True): 沿着最后一个维度计算均值，保持维度不变
        # + self.eps: 加上 epsilon 增加稳定性
        # torch.rsqrt(y): 计算 y 平方根的倒数 (1/sqrt(y))
        # x * ...: 将输入 x 乘以归一化系数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        应用 RMSNorm。
        Args:
            x (torch.Tensor): 输入张量。
        Returns:
            torch.Tensor: 归一化后的张量。
        """
        # 为了数值精度，先将输入转为 float32 进行归一化计算
        normalized_x = self._norm(x.float())
        # 将计算结果转回输入 x 的原始数据类型，然后乘以可学习的权重 weight
        return self.weight * normalized_x.type_as(x)


# 预计算 RoPE (旋转位置嵌入) 的频率因子 (复数形式)
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算 Rotary Positional Embeddings (RoPE) 所需的复数频率 (cisoid)。
    原理：RoPE 通过根据位置旋转特征对来编码绝对位置信息，而不是添加位置向量。
         它将位置信息乘性地引入注意力机制。

    Args:
        dim (int): 需要应用 RoPE 的特征维度 (通常是 head_dim)。
        end (int): 需要预计算频率的最大序列长度。
        theta (float): RoPE 的超参数，控制位置编码的波长范围。

    Returns:
        torch.Tensor: 形状为 (end, dim // 2) 的复数张量，
                      代表每个位置的旋转因子 (cis = cos + i*sin)。
    """
    # 计算频率，公式为: 1.0 / (theta^(2k / dim))，其中 k 是频率索引
    # torch.arange(0, dim, 2): 生成 [0, 2, ..., dim-2] 的索引
    # [: (dim // 2)]: 确保得到 dim // 2 个频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置索引 [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)  # type: ignore

    # 计算位置 `t` 和频率 `freqs` 的外积。
    # 结果形状: (end, dim // 2)。每个元素是 pos * freq，代表每个位置在每个频率上的旋转角度。
    freqs = torch.outer(t, freqs).float()  # type: ignore

    # 使用欧拉公式将角度 (freqs) 转换为复数: e^(i*angle) = cos(angle) + i*sin(angle)。
    # torch.polar(magnitude, angle) 从幅度和角度创建复数。这里幅度为 1。
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 类型
    return pos_cis


# 将旋转位置嵌入 (RoPE) 应用到 Query 和 Key 张量上
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, pos_cis: torch.Tensor):
    """
    将预计算的 RoPE 旋转因子应用到 Query (xq) 和 Key (xk) 张量上。
    原理：通过复数乘法实现特征旋转。将 head_dim 视为 dim/2 个复数，然后乘以位置对应的旋转因子。

    Args:
        xq (torch.Tensor): Query 张量 (Batch, SeqLen, Heads, HeadDim)。
        xk (torch.Tensor): Key 张量 (Batch, SeqLen, Heads, HeadDim)。
        pos_cis (torch.Tensor): 预计算的复数旋转因子 (SeqLen, HeadDim // 2)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 经过旋转的 Query 和 Key 张量。
    """
    def unite_shape(pos_cis: torch.Tensor, x: torch.Tensor):
        """ 辅助函数：调整 pos_cis 的形状，使其能与 x 进行广播运算。"""
        ndim = x.ndim
        assert 0 <= 1 < ndim  # 确保 x 至少有 Batch 和 SeqLen 两个维度
        # pos_cis 的期望形状: (SeqLen, HeadDim // 2)
        # 其中 SeqLen 匹配 x 的维度 1，HeadDim // 2 匹配复数视图下 x 的最后一个维度。
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        # 创建一个新形状，使得 pos_cis 在 Batch (dim 0) 和 Heads (dim 2, 如果存在) 维度上为 1，
        # 以便广播。例如，对于 (B, S, H, D//2) 的 x，pos_cis 形状变为 (1, S, 1, D//2)。
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 1. 将 xq 和 xk 的最后一个维度 (HeadDim) 视为 HeadDim // 2 对实数 (-1, 2)。
    # 2. 使用 view_as_complex 将这些实数对看作复数。
    #    形状变为 (Batch, SeqLen, Heads, HeadDim // 2)。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 pos_cis 的形状以匹配 xq_ 和 xk_ 进行广播。
    pos_cis = unite_shape(pos_cis, xq_)

    # 3. 应用旋转：通过复数乘法 (x + iy) * (cos + i*sin) 实现。
    # 4. 使用 view_as_real 将结果转回实数对。
    # 5. 使用 flatten(3) 将最后两个维度 (HeadDim // 2, 2) 合并回 HeadDim。
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)

    # 6. 将结果转回原始数据类型。
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 重复 Key 和 Value 张量以支持 Grouped Query Attention (GQA) 或 Multi-Query Attention (MQA)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    在 Grouped Query Attention (GQA) 或 Multi-Query Attention (MQA) 中，
    Key 和 Value 的头数 (n_kv_heads) 可能少于 Query 的头数 (n_heads)。
    此函数将 K/V 头重复 `n_rep` 次，使其数量与 Q 头匹配。
    `n_rep` = `n_heads` // `n_kv_heads`。

    等价于特定维度顺序下的 `torch.repeat_interleave(x, dim=2, repeats=n_rep)`。

    Args:
        x (torch.Tensor): 输入的 Key 或 Value 张量，形状为 (Batch, SeqLen, n_kv_heads, HeadDim)。
        n_rep (int): 每个 K/V 头需要重复的次数。

    Returns:
        torch.Tensor: 重复后的张量，形状为 (Batch, SeqLen, n_heads, HeadDim)。
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:  # 如果 n_rep 是 1，说明 Q 和 K/V 头数相同 (MHA)，无需重复
        return x
    # 1. 在 K/V 头维度后插入一个新维度: (bs, slen, n_kv_heads, 1, head_dim)
    # 2. 使用 expand 沿着新维度复制 n_rep 次 (共享内存): (bs, slen, n_kv_heads, n_rep, head_dim)
    # 3. 使用 reshape 合并 K/V 头和重复的维度: (bs, slen, n_kv_heads * n_rep, head_dim)
    #    结果中的 n_kv_heads * n_rep 就是 n_heads。
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 注意力机制模块
class Attention(nn.Module):
    """
    实现多头注意力机制 (Multi-Head Attention, MHA)，
    并支持分组查询注意力 (Grouped Query Attention, GQA) 或多查询注意力 (Multi-Query Attention, MQA)，
    以及可选的 Flash Attention 优化。
    """
    def __init__(self, args: LMConfig):
        """
        初始化 Attention 模块。
        Args:
            args (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        # K/V 头的数量。如果未指定 n_kv_heads，则等于 n_heads (即 MHA)。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保 Q 头数是 K/V 头数的整数倍
        assert args.n_heads % self.n_kv_heads == 0
        # 本地 (单个 GPU 或进程) 的 Q 头数
        self.n_local_heads = args.n_heads
        # 本地 (单个 GPU 或进程) 的 K/V 头数
        self.n_local_kv_heads = self.n_kv_heads
        # 重复因子 n_rep = n_heads / n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads
        # 线性层：将输入映射到 Q, K, V 空间
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 线性层：将注意力输出映射回原始维度
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # Dropout 层，用于 Attention score 和残差连接输出
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout # 存储 dropout 率供 Flash Attention 使用

        # 检查是否可以使用 Flash Attention (需要 PyTorch >= 2.0 且用户启用)
        # Flash Attention 是一种内存高效且快速的注意力实现
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # if not self.flash:
        #     print("警告: 正在使用较慢的注意力实现。Flash Attention 需要 PyTorch >= 2.0。")

        # 预计算因果 MASK (上三角矩阵)，阻止未来信息泄露
        # 形状: (1, 1, max_seq_len, max_seq_len)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1) # 将对角线及以下的元素设为 0，以上设为 -inf
        # 将 mask 注册为 buffer，这样它会被模型保存和加载，但不参与梯度计算
        self.register_buffer("mask", mask, persistent=False) # persistent=False (PyTorch >= 1.9) 可选，指示不保存到 state_dict

    def forward(self,
                x: torch.Tensor, # 输入张量 (Batch, SeqLen, Dim)
                pos_cis: torch.Tensor, # 预计算的 RoPE 旋转因子 (SeqLen, HeadDim // 2)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 上一步的 K/V 缓存 (for generation)
                use_cache=False # 是否使用/返回 K/V 缓存 (for generation)
               ):
        """
        前向传播函数。
        """
        bsz, seq_len, _ = x.shape # 获取 Batch Size 和序列长度

        # 1. 线性投影：计算 Query, Key, Value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape: 将 Q, K, V 分割成多头形式
        # (Batch, SeqLen, Dim) -> (Batch, SeqLen, n_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # (Batch, SeqLen, Dim_kv) -> (Batch, SeqLen, n_kv_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. 应用 RoPE 旋转位置嵌入
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 4. KV Cache 实现 (用于推理加速)
        if past_key_value is not None:
            # 如果传入了过去的 K/V 缓存，将当前的 K, V 拼接到缓存后面
            # past_key_value[0] 是过去的 K, past_key_value[1] 是过去的 V
            # 拼接发生在序列长度维度 (dim=1)
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果 use_cache 为 True，保存当前的 K/V (或拼接后的 K/V) 以供下一步使用
        past_kv = (xk, xv) if use_cache else None

        # 5. GQA/MQA: 重复 K/V 头以匹配 Q 头数量
        # xk/xv: (bsz, cache_len + seq_len, n_local_kv_heads, head_dim)
        # -> (bsz, cache_len + seq_len, n_local_heads, head_dim) after repeat_kv
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 6. Transpose: 调整维度顺序以进行 Batch Matmul
        # (Batch, SeqLen, n_heads, head_dim) -> (Batch, n_heads, SeqLen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 7. 计算 Attention
        if self.flash and seq_len != 1: # Flash Attention 对单 token 推理可能无优化或不支持
            # 使用 PyTorch 内置的 scaled_dot_product_attention (Flash Attention 实现)
            # 优点: 内存高效，速度快，自动处理 mask 和 softmax
            dropout_p = self.dropout if self.training else 0.0 # 训练时应用 dropout
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None, # 对于自回归任务，设置 is_causal=True 即可，无需显式 mask
                dropout_p=dropout_p,
                is_causal=True # 自动应用因果 MASK
            )
        else:
            # 手动计算 Attention (标准方法)
            # (Batch, n_heads, SeqLen_q, head_dim) @ (Batch, n_heads, head_dim, SeqLen_k)
            # -> (Batch, n_heads, SeqLen_q, SeqLen_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # 计算 QK^T / sqrt(d_k)

            # 应用因果 MASK，确保当前位置只能关注之前的位置
            # mask 的形状是 (1, 1, max_len, max_len)，通过切片获取当前序列长度的部分
            # [:seq_len, :seq_len] 适用于 Q 和 K 序列长度相同的情况
            current_seq_len = xk.shape[-1] # 获取 Key 的实际序列长度 (考虑了 KV 缓存)
            scores += self.mask[:, :, :seq_len, :current_seq_len] # 加上 MASK (-inf 会使 softmax 后概率为 0)

            # 计算 Softmax 得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # 转 float 计算 softmax 提高精度
            # 应用 Attention Dropout
            scores = self.attn_dropout(scores)
            # (Batch, n_heads, SeqLen_q, SeqLen_k) @ (Batch, n_heads, SeqLen_k, head_dim)
            # -> (Batch, n_heads, SeqLen_q, head_dim)
            output = scores @ xv # 使用权重加权 V

        # 8. Reshape and Output Projection
        # (Batch, n_heads, SeqLen, head_dim) -> (Batch, SeqLen, n_heads, head_dim)
        output = output.transpose(1, 2).contiguous() # contiguous() 确保内存连续，有时 reshape 需要
        # (Batch, SeqLen, n_heads, head_dim) -> (Batch, SeqLen, n_heads * head_dim)
        output = output.reshape(bsz, seq_len, -1)
        # 应用输出线性层和残差 Dropout
        output = self.resid_dropout(self.wo(output))
        return output, past_kv # 返回注意力输出和更新后的 K/V 缓存


# 前馈神经网络 (Feed Forward Network) 模块
class FeedForward(nn.Module):
    """
    实现 Transformer 块中的位置相关前馈网络 (FFN)。
    通常包含两个线性层和一个非线性激活函数。这里使用 SwiGLU 变体。
    原理：对每个位置的表示进行独立的非线性变换，增加模型的表达能力。
    结构 (SwiGLU): x -> Linear1(x), Linear3(x) -> SiLU(Linear1(x)) * Linear3(x) -> Linear2 -> Dropout
    """
    def __init__(self, config: LMConfig):
        """
        初始化 FeedForward 模块。
        Args:
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        # 如果未指定隐藏层维度 hidden_dim，则按 LLaMA 的方式计算
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim # 基础维度通常是 4 倍
            hidden_dim = int(2 * hidden_dim / 3) # LLaMA 使用 2/3 缩放
            # 确保 hidden_dim 是 multiple_of 的整数倍 (为了硬件优化)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # 定义三个线性层 (对应 SwiGLU 结构)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False) # Gate projection
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False) # Down projection
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False) # Up projection
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量 (Batch, SeqLen, Dim)。
        Returns:
            torch.Tensor: FFN 输出张量 (Batch, SeqLen, Dim)。
        """
        # SwiGLU 实现: silu(w1(x)) * w3(x)
        swiglu_out = F.silu(self.w1(x)) * self.w3(x)
        # 应用第二个线性层 w2 和 dropout
        return self.dropout(self.w2(swiglu_out))


# MoE (Mixture of Experts) 的门控网络 (Gating Network)
class MoEGate(nn.Module):
    """
    实现 MoE 层的门控网络 (Gating Network)。
    原理：为每个输入 token 计算一组分数 (logits)，决定将该 token 路由到哪些 Expert FFN。
         通常选择分数最高的 Top-K 个 Expert。
         还会计算一个辅助损失 (Auxiliary Loss) 来鼓励 Expert 负载均衡。
    """
    def __init__(self, config: LMConfig):
        """
        初始化 MoE 门控网络。
        Args:
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        self.config = config
        # 每个 token 选择的 Expert 数量
        self.top_k = config.num_experts_per_tok
        # 总共可路由的 Expert 数量
        self.n_routed_experts = config.n_routed_experts

        # Gating 分数计算方式 (目前只支持 softmax)
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数
        self.alpha = config.aux_loss_alpha
        # 是否在序列维度上计算辅助损失 (默认为 False，在 Batch 维度上计算)
        self.seq_aux = config.seq_aux

        # 是否对 Top-K 的概率进行归一化
        self.norm_topk_prob = config.norm_topk_prob
        # Gating 网络的输入维度 (等于模型维度)
        self.gating_dim = config.dim
        # 门控网络的权重参数 (可学习)，形状为 (专家数量, 输入维度)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ 初始化门控网络的权重。 """
        import torch.nn.init as init
        # 使用 Kaiming 均匀分布初始化，适合 ReLU 类激活函数 (这里虽然是 softmax，但也是常用初始化)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播函数。
        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态 (Batch, SeqLen, Dim)。
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - topk_idx: 每个 token 被分配到的 Top-K Expert 的索引 (Batch*SeqLen, top_k)。
                - topk_weight: 每个 token 分配给对应 Top-K Expert 的权重 (Batch*SeqLen, top_k)。
                - aux_loss: 负载均衡辅助损失 (标量)。
        """
        bsz, seq_len, h = hidden_states.shape
        # 将输入 reshape 为 (Batch*SeqLen, Dim) 以便进行线性变换
        hidden_states = hidden_states.view(-1, h)
        # 计算门控 Logits: (Batch*SeqLen, Dim) @ (Dim, n_routed_experts) -> (Batch*SeqLen, n_routed_experts)
        logits = F.linear(hidden_states, self.weight, None)

        # 根据 scoring_func 计算门控分数 (目前仅支持 softmax)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'不支持的 MoE 门控评分函数: {self.scoring_func}')

        # 选择 Top-K 的 Expert 及其对应的分数(权重)
        # topk_weight: (Batch*SeqLen, top_k) 记录每个 token 分配给 Expert 的权重 形状 为 (Batch*SeqLen, top_k)
        # topk_idx: (Batch*SeqLen, top_k) 记录每个 token 被分配到的 Expert 的索引 形状 为 (Batch*SeqLen, top_k)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 可选：对 Top-K 的权重进行归一化，使它们的和为 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20 # 加 epsilon 防除零
            topk_weight = topk_weight / denominator

        # 计算辅助损失 (仅在训练时且 alpha > 0 时)
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores # 使用原始的 softmax 分数计算辅助损失
            aux_topk = self.top_k # 使用配置的 top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # (bsz, seq_len * top_k)

            if self.seq_aux: # 在序列维度计算 (不常用)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # (bsz, seq_len, n_experts)
                # 计算每个 expert 在每个 batch item 中被选中的比例
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts) # 除以期望的均匀选中次数
                # 辅助损失 = sum(选中比例 * 平均分数) * alpha
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else: # 在 Batch 维度计算 (常用)
                # 计算每个 expert 被选中的频率 (fraction of tokens routed to expert i)
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts) # (bsz * seq_len * top_k, n_experts)
                ce = mask_ce.float().mean(0) # (n_experts,) - 每个 expert 处理的 token 比例
                # 计算每个 expert 的平均路由分数 (average routing probability for expert i)
                Pi = scores_for_aux.mean(0) # (n_experts,)
                # 计算 f_i = N * P_i，其中 N 是专家数，P_i 是选中频率
                fi = ce * self.n_routed_experts
                # 辅助损失 = sum(P_i * f_i) * alpha = sum(平均分数 * 选中频率 * N) * alpha
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device) # 非训练模式或 alpha=0 时，损失为 0

        return topk_idx, topk_weight, aux_loss


# MoE 前馈神经网络 (结合 Gating 和多个 Expert FFN)
class MOEFeedForward(nn.Module):
    """
    实现 Mixture-of-Experts (MoE) 前馈网络层。
    包含一个门控网络 (MoEGate) 和多个 Expert FFN (FeedForward)。
    原理：对于每个 token，门控网络选择 Top-K 个 Expert，然后将 token 输入这些 Expert 计算，
         最终结果是这些 Expert 输出的加权和 (权重由门控网络提供)。
         可以显著增加模型参数量，但计算量只增加少量 (每个 token 只通过 K 个 Expert)。
         可能包含一个共享的 Expert (shared_experts)，所有 token 都会通过它。
    """
    def __init__(self, config: LMConfig):
        """
        初始化 MOEFeedForward 模块。
        Args:
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        self.config = config
        # 创建多个 Expert FFN，数量为 n_routed_experts
        self.experts = nn.ModuleList([
            FeedForward(config) # 每个 Expert 是一个独立的 FFN
            for _ in range(config.n_routed_experts)
        ])
        # 创建门控网络
        self.gate = MoEGate(config)
        # 如果配置了共享 Expert，则创建它
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)
        # 用于存储辅助损失
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量 (Batch, SeqLen, Dim)。
        Returns:
            torch.Tensor: MoE FFN 输出张量 (Batch, SeqLen, Dim)。
        """
        identity = x # 保存原始输入，用于可能的 shared_experts 计算
        orig_shape = x.shape # 保存原始形状
        bsz, seq_len, _ = x.shape

        # 1. 使用门控网络获取路由决策
        # topk_idx: (bsz*seq_len, top_k)
        # topk_weight: (bsz*seq_len, top_k)
        # aux_loss: scalar
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 将辅助损失存储起来，方便外部访问
        self.aux_loss = aux_loss

        # 将输入 reshape 为 (Batch*SeqLen, Dim)
        x = x.view(-1, x.shape[-1])
        # 将 Top-K 索引也 flatten
        flat_topk_idx = topk_idx.view(-1) # (bsz*seq_len*top_k)

        # 2. 根据训练/推理模式选择不同的计算路径
        if self.training:
            # 训练模式：通常使用 token-dropping 或类似方法简化计算
            # 这里实现了一种常见方法：复制输入，然后根据索引路由
            # 将输入 token 复制 top_k 次，因为每个 token 要送入 top_k 个 expert
            # x: (bsz*seq_len, dim) -> (bsz*seq_len*top_k, dim)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个空的输出张量用于存储 Expert 的结果
            y = torch.empty_like(x, dtype=torch.float16) # 使用 float16 可能为了节省内存

            # 遍历所有 Expert
            for i, expert in enumerate(self.experts):
                # 找到应该由当前 expert 处理的 token 的索引 (在复制后的 x 中的索引)
                expert_mask = (flat_topk_idx == i)
                # 如果没有 token 分配给这个 expert，跳过
                if expert_mask.any():
                    # 选择对应的 token
                    tokens_for_expert = x[expert_mask]
                    # 计算 Expert 输出，并确保数据类型一致
                    expert_output = expert(tokens_for_expert).to(y.dtype)
                    # 将结果放回 y 中对应位置
                    y[expert_mask] = expert_output

            # Expert 输出加权求和
            # y: (bsz*seq_len*top_k, dim) -> (bsz*seq_len, top_k, dim)
            y = y.view(*topk_weight.shape, -1)
            # topk_weight: (bsz*seq_len, top_k) -> (bsz*seq_len, top_k, 1) for broadcasting
            # Weighted sum: (bsz*seq_len, top_k, dim) * (bsz*seq_len, top_k, 1) -> sum over top_k dim
            # -> (bsz*seq_len, dim)
            y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
            # Reshape 回原始形状 (Batch, SeqLen, Dim)
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用优化的 moe_infer 函数，避免复制输入
            # flat_topk_idx: (bsz*seq_len*top_k)
            # topk_weight: (bsz*seq_len, top_k) -> (bsz*seq_len*top_k, 1)
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 3. 如果有共享 Expert，将其输出加到 MoE 输出上
        if hasattr(self, 'shared_experts') and self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad() # 推理时不需要计算梯度
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化的 MoE 推理实现。
        原理：将所有 token 按其分配到的 Expert 排序，然后批量处理每个 Expert 的 token，
             最后将结果分散加回到原始 token 位置。避免了 `repeat_interleave` 的内存开销。

        Args:
            x (torch.Tensor): 输入 tokens (Batch*SeqLen, Dim)。
            flat_expert_indices (torch.Tensor): 所有 token 的 Expert 索引 (Batch*SeqLen*top_k)。
            flat_expert_weights (torch.Tensor): 所有 token 的 Expert 权重 (Batch*SeqLen*top_k, 1)。

        Returns:
            torch.Tensor: MoE 输出 (Batch*SeqLen, Dim)。
        """
        # 创建一个缓存张量，用于累加每个 token 的最终输出
        expert_cache = torch.zeros_like(x)
        # 按照分配的 Expert 索引对所有 (token, expert) 对进行排序
        # idxs 是排序后的原始索引 (指 flat_expert_indices 中的位置)
        idxs = flat_expert_indices.argsort()
        # 计算每个 Expert 需要处理的 token 数量的累积和
        # 例如，如果 expert 0 处理 6 个，expert 1 处理 9 个，expert 2 处理 5 个...
        # tokens_per_expert = [6, 15, 20, ...]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 找到每个排好序的 (token, expert) 对对应的原始 token 索引 (在 x 中的索引)
        # idxs // self.config.num_experts_per_tok 是因为 flat_expert_indices 的长度是 token 数 * top_k
        token_idxs = idxs // self.config.num_experts_per_tok

        # 遍历每个 Expert
        for i, end_idx in enumerate(tokens_per_expert):
            # 确定当前 Expert 处理的 token 范围在排序后列表中的起始和结束索引
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果当前 Expert 没有分配到 token，则跳过
            if start_idx == end_idx:
                continue

            # 获取当前 Expert 模块
            expert = self.experts[i]
            # 获取分配给当前 Expert 的原始 token 的索引 (在 x 中的索引)
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 从 x 中提取这些 token
            expert_tokens = x[exp_token_idx]
            # 通过 Expert 计算输出，并确保数据类型正确
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 将 Expert 输出乘以对应的门控权重
            # flat_expert_weights[idxs[start_idx:end_idx]] 获取排序后对应范围的权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 将加权后的 Expert 输出累加到 expert_cache 中对应的原始 token 位置
            # exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]) 构造用于 scatter_add_ 的索引张量
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


# Transformer Block (单个层)
class MiniMindBlock(nn.Module):
    """
    表示一个完整的 Transformer 层/块。
    包含一个自注意力 (Self-Attention) 子层和一个前馈网络 (FFN 或 MoE FFN) 子层。
    每个子层前后都有残差连接 (Residual Connection) 和层归一化 (Layer Normalization, 这里是 RMSNorm)。
    结构：
    x -> RMSNorm -> Attention -> + -> x_attn
                                     |
    x_attn -> RMSNorm -> FFN/MoE -> + -> output
    """
    def __init__(self, layer_id: int, config: LMConfig):
        """
        初始化 Transformer Block。
        Args:
            layer_id (int): 当前层的 ID (用于调试或特定层操作)。
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        # 注意力子层
        self.attention = Attention(config)
        self.layer_id = layer_id # 存储层 ID
        # 注意力子层前的 RMSNorm
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # FFN/MoE 子层前的 RMSNorm
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # FFN/MoE 子层 (根据配置选择使用普通 FFN 还是 MoE FFN)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量 (Batch, SeqLen, Dim)。
            pos_cis (torch.Tensor): RoPE 旋转因子 (SeqLen, HeadDim // 2)。
            past_key_value (Optional[Tuple]): 上一步的 K/V 缓存。
            use_cache (bool): 是否使用/返回 K/V 缓存。
        Returns:
            Tuple[torch.Tensor, Optional[Tuple]]:
                - out: Transformer Block 的输出张量 (Batch, SeqLen, Dim)。
                - past_kv: 更新后的 K/V 缓存。
        """
        # 1. 注意力子层
        # 应用 RMSNorm，然后通过 Attention 模块，最后加上残差连接
        h_attn, past_kv = self.attention(
            self.attention_norm(x), # 先 Norm
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn # 残差连接

        # 2. FFN/MoE 子层
        # 应用 RMSNorm，然后通过 FeedForward/MOEFeedForward 模块，最后加上残差连接
        ff_out = self.feed_forward(self.ffn_norm(h)) # 先 Norm
        out = h + ff_out # 残差连接

        return out, past_kv


# 完整的语言模型
class MiniMindLM(PreTrainedModel):
    """
    基于 Transformer Block 构建的自回归语言模型 (LM)。
    继承自 Hugging Face 的 PreTrainedModel，方便加载预训练权重和使用 Trainer API。
    结构：
    Input Ids -> Embedding -> Dropout -> N x Transformer Blocks -> Final RMSNorm -> Output Linear Layer -> Logits
    """
    config_class = LMConfig # 指定配置类，Hugging Face 需要

    def __init__(self, params: LMConfig = None):
        """
        初始化 MiniMindLM 模型。
        Args:
            params (LMConfig, optional): 模型配置参数。如果为 None，则使用默认 LMConfig()。
        """
        # 如果 params 未提供，则创建一个默认的 LMConfig 实例
        self.params = params or LMConfig()
        # 调用 PreTrainedModel 的初始化函数，传入配置
        super().__init__(self.params)

        # 存储词汇表大小和层数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # Token Embedding 层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # Embedding 后的 Dropout 层
        self.dropout = nn.Dropout(params.dropout)
        # Transformer 层列表
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        # 最后的 RMSNorm 层 (在输出层之前)
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 输出线性层 (映射到词汇表空间)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 权重绑定 (Weight Tying): Embedding 层和输出层的权重共享，可以减少参数量并提高性能
        self.tok_embeddings.weight = self.output.weight

        # 预计算并注册 RoPE 旋转因子作为 buffer
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, # head_dim
                                                theta=params.rope_theta,
                                                end=params.max_seq_len * 2 # 预计算长度可以设置大一些
                                                ),
                             persistent=False) # 不保存到 state_dict

        # 初始化 Hugging Face 标准输出对象
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, # 输入 token ID (Batch, SeqLen)
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # 过去的 K/V 缓存列表
                use_cache: bool = False, # 是否使用/返回 K/V 缓存
                logits_to_keep: Union[int, torch.Tensor] = 0, # 仅计算最后几个 token 的 logits (推理优化)
                **args):
        """
        模型的前向传播函数。
        Args:
            input_ids: 输入的 token ID。
            past_key_values: 之前时间步的 Key/Value 缓存，用于加速生成。列表长度等于层数，每个元素是 (Key, Value) 元组。
            use_cache: 是否返回更新后的 past_key_values。
            logits_to_keep: 如果是正整数 n，则只计算并返回最后 n 个 token 的 logits。如果是 Tensor，则根据 Tensor 中的索引计算。0 表示计算所有 logits。
            **args: 其他参数，例如 'start_pos' 用于生成时指定 RoPE 的起始位置。
        Returns:
            CausalLMOutputWithPast: 包含 logits, past_key_values, last_hidden_state, aux_loss 的对象。
        """
        # 如果未提供 past_key_values，初始化为 None 列表
        past_key_values = past_key_values or [None] * len(self.layers)
        # 获取 RoPE 的起始位置，默认为 0。在生成时，会根据已生成长度设置。
        start_pos = args.get('start_pos', 0)

        # 1. Token Embeddings and Dropout
        h = self.dropout(self.tok_embeddings(input_ids)) # (Batch, SeqLen, Dim)

        # 2. 获取当前输入序列对应的 RoPE 旋转因子
        # 从预计算的 pos_cis 中切片出需要的长度
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]

        # 3. 通过 Transformer Layers
        past_kvs = [] # 用于存储每一层更新后的 K/V 缓存
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l], # 传入该层的 K/V 缓存
                use_cache=use_cache
            )
            past_kvs.append(past_kv) # 收集更新后的缓存

        # 4. Final Norm and Output Layer
        # 根据 logits_to_keep 确定需要计算 logits 的 hidden state 部分
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            # 如果是整数 n > 0，只取最后 n 个时间步的 hidden state
            slice_indices = slice(-logits_to_keep, None)
        elif isinstance(logits_to_keep, torch.Tensor):
            # 如果是 Tensor，则根据 Tensor 中的索引选取 hidden state (更灵活，但可能不常用)
             slice_indices = logits_to_keep
        else: # 默认情况 (0 or False)，取所有时间步
            slice_indices = slice(None, None)

        # 应用最后的 RMSNorm 和输出线性层
        h_norm = self.norm(h)
        logits = self.output(h_norm[:, slice_indices, :]) # 只计算选定部分的 logits

        # 5. 计算 MoE 辅助损失 (如果使用了 MoE 层)
        # 累加所有 MoE 层产生的辅助损失
        aux_loss_list = [l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward)]
        aux_loss = sum(aux_loss_list) if aux_loss_list else None # 如果没有 MoE 层，则为 None

        # 6. 组装输出对象
        # 将计算结果填充到 CausalLMOutputWithPast 对象中
        self.OUT.__setitem__('last_hidden_state', h) # 最后一个 Transformer 层的输出
        self.OUT.__setitem__('logits', logits)       # 计算得到的 Logits
        self.OUT.__setitem__('aux_loss', aux_loss)   # MoE 辅助损失
        self.OUT.__setitem__('past_key_values', past_kvs if use_cache else None) # 返回的 K/V 缓存

        return self.OUT

    @torch.inference_mode() # 推理时使用，禁用梯度计算，节省内存和计算
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, num_return_sequences=1, **args):
        """
        文本生成函数。
        Args:
            input_ids (torch.Tensor): 输入的 prompt token IDs (Batch, PromptSeqLen)。
            eos_token_id (int): 结束符 token ID。
            max_new_tokens (int): 要生成的最大新 token 数量。
            temperature (float): 温度系数，控制生成文本的随机性。越小越确定。
            top_p (float): Top-p (Nucleus) sampling 的阈值。仅考虑累积概率达到 p 的 token。
            stream (bool): 是否以流式 (generator) 方式返回生成的 token。
            rp (float): 重复惩罚 (Repetition Penalty) 系数。> 1 时降低已出现 token 的概率。
            use_cache (bool): 是否使用 K/V 缓存加速生成。
            pad_token_id (int): 用于填充的 token ID。
            num_return_sequences (int): 每个输入 prompt 要生成的序列数量。
            **args: 传递给 forward 函数的其他参数。
        Returns:
            Union[torch.Tensor, Iterator[torch.Tensor]]:
                - 如果 stream=False: 返回包含完整生成序列的 Tensor (Batch*num_return_sequences, TotalSeqLen)。
                - 如果 stream=True: 返回一个生成器，每次 yield 新生成的 token (Batch, NewTokensAtStep)。
        """
        # 流式生成：直接调用内部的 _stream 生成器
        if stream:
            # 注意：流式生成通常只处理 batch size=1 的情况，且 num_return_sequences=1
            # 如果需要处理 batch 或多序列，需要在此处添加循环或修改 _stream
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 非流式生成 (直接生成完整序列)
        generated = [] # 存储每个生成的完整序列
        # 遍历输入的每个 prompt
        for i in range(input_ids.size(0)):
            # 提取非填充部分的 prompt
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0) # (1, PromptSeqLen)
            # 为每个 prompt 生成 num_return_sequences 个序列
            for _ in range(num_return_sequences):
                # 调用 _stream 生成器获取生成过程
                out_stream = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
                # 从生成器中收集所有生成的 token 片段
                # 每个 tokens 是 (1, GeneratedTokensSoFar)
                # tokens[:, -1:] 取最后一个 token (1, 1)
                tokens_list = [tokens[:, -1:] for tokens in out_stream]
                # 将所有新生成的 token 拼接起来
                gen = torch.cat(tokens_list, dim=-1) if tokens_list else torch.empty((1, 0), dtype=non_pad.dtype, device=non_pad.device)
                # 将 prompt 和生成的部分拼接成完整序列
                full_sequence = torch.cat([non_pad, gen], dim=-1) # (1, TotalSeqLen)
                generated.append(full_sequence)

        # 对所有生成的序列进行填充，使它们长度一致
        max_length = max(seq.size(1) for seq in generated) if generated else 0
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        # 将列表中的所有序列张量合并成一个大的张量
        output = torch.cat(generated, dim=0) if generated else torch.empty((0, max_length), dtype=input_ids.dtype, device=input_ids.device)
        # 调整形状以匹配 (Batch*num_return_sequences, TotalSeqLen)
        res = output.view(input_ids.size(0) * num_return_sequences, -1)
        return res

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """
        内部的自回归生成循环，实现为生成器 (generator)。
        每次迭代生成一个 token 并 yield。
        """
        prompt_len = input_ids.shape[1] # 初始 prompt 长度
        total_len = prompt_len + max_new_tokens # 目标总长度
        first_seq, past_kvs = True, None # 标记是否是第一次输入，初始化 K/V 缓存为 None

        # 循环生成新 token，直到达到最大长度
        while input_ids.shape[1] < total_len:
            # --- KV Cache 处理 ---
            if first_seq or not use_cache:
                # 第一次输入或不使用缓存：将整个 input_ids 输入模型
                current_input_ids = input_ids
                start_pos = 0 # RoPE 起始位置为 0
                first_seq = False # 不再是第一次输入
            else:
                # 使用缓存：只将上一步生成的最后一个 token 输入模型
                current_input_ids = input_ids[:, -1:] # (Batch, 1)
                # RoPE 起始位置是当前序列长度减 1
                start_pos = input_ids.shape[1] - 1
            # --- 模型前向传播 ---
            # 调用 forward 函数获取 logits 和更新的 K/V 缓存
            # logits_to_keep=1: 优化，只计算最后一个 token 的 logits
            out = self(current_input_ids, past_key_values=past_kvs, use_cache=use_cache,
                       start_pos=start_pos, logits_to_keep=1, **args)
            # 提取最后一个 token 的 logits 和更新后的 K/V 缓存
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values # logits shape: (Batch, VocabSize)

            # --- Logits 修改 ---
            # 1. 重复惩罚 (Repetition Penalty)
            # 仅在 rp > 1 时应用
            if rp != 1.:
                # 获取当前已生成的所有 token ID (不包括 Batch 维度)
                seen_tokens = set(input_ids.tolist()[0]) # 假设 Batch Size=1
                # 对 logits 中对应已出现 token 的位置进行惩罚 (除以 rp)
                for token_id in seen_tokens:
                     logits[:, token_id] /= rp
                    # 注意: 更常见的实现是对正 logits 除以 rp，对负 logits 乘以 rp

            # 2. 温度缩放 (Temperature Scaling)
            # logits 除以温度，使得概率分布更平缓 (temp > 1) 或更尖锐 (temp < 1)
            logits.div_(temperature + 1e-9) # 加 epsilon 防止除以零

            # 3. Top-p (Nucleus) Sampling
            if top_p is not None and top_p < 1.0:
                # 对 logits 排序并计算 softmax 概率
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 找到累积概率首次超过 top_p 的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将移除标记向右移动一位，确保至少保留第一个 token (概率最高的)
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                # 将移除标记映射回原始索引
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # 将要移除的 token 的 logits 设置为负无穷，使其在 softmax 后概率为 0
                logits[indices_to_remove] = -float('Inf')

            # --- Token 采样 ---
            # 从修改后的 logits 计算概率分布并采样
            probs = F.softmax(logits, dim=-1)
            input_ids_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)

            # --- 更新序列并 Yield ---
            # 将新生成的 token 拼接到当前序列后面
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            # Yield 新生成的 token 部分 (从 prompt 之后开始)
            yield input_ids[:, prompt_len:] # (Batch, NewTokensSoFar)

            # --- 终止条件 ---
            # 如果生成的 token 是结束符，则停止生成
            # .item() 适用于 Batch Size=1 的情况
            if input_ids_next.item() == eos_token_id:
                break
            # 如果序列长度达到最大值，循环也会在下一次迭代开始时终止