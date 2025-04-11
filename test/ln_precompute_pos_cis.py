import torch
import math

# --- 定义 precompute_pos_cis 函数 (与你提供的源码一致) ---
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算 RoPE 所需的复数旋转因子 (cis = cos + i*sin)。

    Args:
        dim (int): 每个注意力头的维度 (head_dim)。
        end (int): 最大序列长度。
        theta (float): RoPE 的基准参数。

    Returns:
        torch.Tensor: 形状为 (end, dim // 2) 的复数张量。
    """
    # 1. 计算基础旋转频率 θ_j = 1.0 / (theta^(2j/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # freqs shape: [dim // 2]

    # 2. 创建位置索引 t = [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t shape: [end]

    # 3. 计算旋转角度 m * θ_j (通过外积)
    # freqs variable reused to store angles
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # freqs (angles) shape: [end, dim // 2]

    # 4. 将角度转换为复数 cos(angle) + i*sin(angle)
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # pos_cis shape: [end, dim // 2]

    return pos_cis

# --- 示例参数 ---
example_dim = 4       # 假设注意力头维度为 4
example_end = 3       # 假设最大序列长度为 3 (位置 0, 1, 2)
example_theta = 10000.0 # 使用常用的 theta 值

print(f"--- Example Parameters ---")
print(f"Head Dimension (dim): {example_dim}")
print(f"Max Sequence Length (end): {example_end}")
print(f"Theta: {example_theta}")
print("-" * 20)

# --- 手动计算步骤 (用于理解) ---
print("\n--- Manual Calculation Steps ---")

# Step 1: Calculate base frequencies (θ_j)
print("Step 1: Calculate base frequencies (θ_j)")
dim_half = example_dim // 2
indices_j = torch.arange(0, example_dim, 2)[:dim_half].float() # Indices 2*j: [0., 2.]
exponents = indices_j / example_dim                         # Exponents 2j/dim: [0./4, 2./4] = [0., 0.5]
theta_powers = example_theta ** exponents                     # theta^(2j/dim): [10000^0, 10000^0.5] = [1., 100.]
step1_freqs = 1.0 / theta_powers                              # θ_j = 1 / theta^(2j/dim): [1./1., 1./100.] = [1., 0.01]
print(f"  Base frequencies (θ_j): {step1_freqs}")             # Shape: [dim//2] = [2]

# Step 2: Create position indices (t)
print("\nStep 2: Create position indices (t)")
step2_t = torch.arange(example_end)                         # t = [0, 1, 2]
print(f"  Position indices (t): {step2_t}")                  # Shape: [end] = [3]

# Step 3: Calculate rotation angles (m * θ_j) using outer product
print("\nStep 3: Calculate rotation angles (m * θ_j)")
step3_angles = torch.outer(step2_t, step1_freqs).float()
# Outer product of [0, 1, 2] and [1., 0.01]:
# [[0*1, 0*0.01], [1*1, 1*0.01], [2*1, 2*0.01]]
# = [[0.0, 0.0], [1.0, 0.01], [2.0, 0.02]]
print(f"  Rotation angles matrix (shape: [end, dim // 2]):\n{step3_angles}") # Shape: [3, 2]

# Step 4: Convert angles to complex numbers (cos(angle) + i*sin(angle))
print("\nStep 4: Convert angles to complex numbers (pos_cis)")
step4_pos_cis_manual = torch.polar(torch.ones_like(step3_angles), step3_angles)
print(f"  Resulting complex numbers (shape: [end, dim // 2]):\n{step4_pos_cis_manual}") # Shape: [3, 2]
print("-" * 20)

# --- 调用函数并打印结果 ---
print("\n--- Function Call Result ---")
pos_cis_result = precompute_pos_cis(
    dim=example_dim,
    end=example_end,
    theta=example_theta
)

print(f"Output tensor shape: {pos_cis_result.shape}")
print(f"Output tensor dtype: {pos_cis_result.dtype}")
print(f"Output tensor (pos_cis):\n{pos_cis_result}")
print("-" * 20)

# --- 验证手动计算和函数结果是否一致 ---
print("\n--- Verification ---")
are_close = torch.allclose(step4_pos_cis_manual, pos_cis_result)
print(f"Manual calculation matches function output: {are_close}")