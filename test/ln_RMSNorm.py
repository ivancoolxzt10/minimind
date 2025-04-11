import torch
from torch import nn

# --- 重现实代码中的 RMSNorm 类 ---
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 为简单起见，例子中我们假设输入已经是 float 类型，省略 .float() 和 .type_as(x)
        # return self.weight * self._norm(x.float()).type_as(x)
        return self.weight * self._norm(x)

# --- 示例数据 ---
dim = 4
eps = 1e-6

# 假设输入 x (Batch=1, Seq=1, Dim=4)
x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], dtype=torch.float32)
print(f"Input x: {x}")
print(f"Input shape: {x.shape}")

# 初始化 RMSNorm 层
rms_norm = RMSNorm(dim=dim, eps=eps)

# 假设可学习的 weight 被初始化为 [1.0, 1.1, 0.9, 1.0] (为了演示效果，不全是1)
initial_weight = torch.tensor([1.0, 1.1, 0.9, 1.0], dtype=torch.float32)
# 手动设置 weight (在实际训练中它是通过 nn.Parameter 自动管理的)
with torch.no_grad():
    rms_norm.weight.copy_(initial_weight)
print(f"Learnable weight: {rms_norm.weight}")

# --- 手动计算 RMSNorm ---
print("\n--- Manual Calculation ---")

# 1. 计算 x 的平方
x_squared = x.pow(2)
print(f"1. x^2: {x_squared}") # [[[ 1.,  4.,  9., 16.]]]

# 2. 计算平方的均值 (沿最后一个维度 dim=-1)
mean_squared = x_squared.mean(-1, keepdim=True)
print(f"2. Mean(x^2): {mean_squared}") # [[[ 7.5 ]]]  ( (1+4+9+16) / 4 = 30 / 4 = 7.5 )

# 3. 加上 epsilon
mean_squared_eps = mean_squared + eps
print(f"3. Mean(x^2) + eps: {mean_squared_eps}") # [[[ 7.500001 ]]] (近似)

# 4. 计算 RMS 值 (均方根)
rms = torch.sqrt(mean_squared_eps)
print(f"4. RMS = sqrt(Mean(x^2) + eps): {rms}") # [[[ 2.738613 ]]] (sqrt(7.5))

# 5. 归一化 x ( x / RMS )
normalized_x = x / rms
print(f"5. Normalized x = x / RMS: {normalized_x}")
# [[[ 0.3651, -0.7303,  1.0954, -1.4606 ]]] (例如: 1.0 / 2.7386 = 0.3651)

# 6. 乘以可学习的 weight
manual_output = normalized_x * rms_norm.weight # 使用广播机制
print(f"6. Manual Output = Normalized x * weight: {manual_output}")
# [[[ 0.3651, -0.8033,  0.9859, -1.4606 ]]] (例如: -0.7303 * 1.1 = -0.8033)

# --- 使用 RMSNorm 类计算 ---
print("\n--- Using RMSNorm Class ---")
class_output = rms_norm(x)
print(f"Class Output: {class_output}")

# --- 验证 ---
print("\n--- Verification ---")
are_close = torch.allclose(manual_output, class_output, atol=1e-5) # 检查两个输出是否足够接近
print(f"Manual calculation matches class output: {are_close}")