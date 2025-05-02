import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# size (cm)
Lx = 15.0  # 宽
Ly = 12.0  # 高

# 网格数
Nx = 75
Ny = 60

# 间距ΔX, ΔY
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
beta = dx / dy 

# 松弛因子ω
omega = 1.8

# 判定收敛
tolerance = 1e-6
max_iter = 10000

# 初值
T = np.full((Nx, Ny), 20.0)  # initial guess = 20°C everywhere

# 边界条件
T[:, -1] = 100.0  # 上边
T[:, 0] = 20.0    # 下边
T[0, :] = 20.0    # 左边
T[-1, :] = 20.0   # 右边

# SOR迭代
for iteration in range(max_iter):
    T_old = T.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            T_new = (1 / (2 * (1 + beta**2))) * (
                T[i+1, j] + T[i-1, j] + beta**2 * (T[i, j+1] + T[i, j-1])
            )
            T[i, j] += omega * (T_new - T[i, j])
    error = np.max(np.abs(T - T_old))
    if error < tolerance:
        print(f"经过{iteration+1} 次迭代收敛，松弛因子 ω = {omega}")
        break
else:
    print("超过最大次数")

# 可视化
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

plt.figure(figsize=(8, 5))
cp = plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(cp, label='Temperature (°C)')
plt.title(f"2D Heat Conduction (SOR, ω = {omega})")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.tight_layout()
plt.show()
# --- Plotting 3D Surface ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot surface and capture the handle
surf = ax.plot_surface(X, Y, T, cmap='hot', edgecolor='k', linewidth=0.2)

# Add colorbar using the surface handle
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, label='Temperature (°C)')
ax.set_title("3D Temperature Distribution on Plate (SOR)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("Temperature (°C)")

plt.tight_layout()
plt.show()