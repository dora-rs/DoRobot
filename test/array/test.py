import numpy as np

# 假设原数组 shape 是 (H, W)
arr = np.random.rand(3, 4)  # 例如 H=5, W=10
print("原 shape:", arr.shape)  # (5, 10)
print("原 arr:", arr)  # (5, 10)

# 方法1: 使用 np.expand_dims
arr_expanded = np.expand_dims(arr, axis=-1)  # 在最后添加一个维度
print("expand_dims 后 shape:", arr_expanded.shape)  # (5, 10, 1)
print("expand_dims 后 arr:", arr_expanded)

# 方法2: 使用 arr[:, :, np.newaxis]
arr_expanded = arr[:, :, np.newaxis]
print("newaxis 后 shape:", arr_expanded.shape)  # (5, 10, 1)
print("newaxis 后 arr:", arr_expanded)

# 方法3: 使用 reshape
arr_expanded = arr.reshape(*arr.shape, 1)  # 或 arr.reshape(arr.shape + (1,))
print("reshape 后 shape:", arr_expanded.shape)  # (5, 10, 1)
print("reshape 后 arr:", arr_expanded)
