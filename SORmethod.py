import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# SOR迭代
def SOR(T, omega, beta, max_iter, tolerance):
    Nx, Ny = T.shape
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
    return iteration+1
def find_best_omega(omega_values, T, beta, max_iter, tolerance, results):
    for omega in omega_values:
        if omega in results:          # 已算过的不重复
            continue
        Ti = T.copy()
        iterations = SOR(Ti, omega, beta, max_iter, tolerance)
        results[omega] = iterations
        w_best= min(results, key=results.get)
    return w_best
# 可视化
def draw(Lx,Ly,Nx,Ny,T_final,best_omega):
    plt.figure(figsize=(8, 5))
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cp = plt.contourf(X, Y, T_final, 50, cmap='hot')
    plt.colorbar(cp, label='Temperature (°C)')
    plt.title(f"2D Heat Conduction (SOR, ω = {best_omega},Nx={Nx},Ny={Ny})")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.tight_layout()
    plt.show()

def maincode(Nx,Ny):
    # 网格划分
    Lx = 15.0  # 宽
    Ly = 12.0  # 高
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    beta = dx / dy 
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
    results = {} #存储每个ω的迭代次数

    #设定松弛因子搜索范围
    left=1.0
    right=2.0
    step=0.2
    omega_values = np.arange(left,right,step) 
    min_iterations_old = max_iter
    for _ in tqdm(range(5)):
      w_best=find_best_omega(omega_values, T, beta, max_iter, tolerance, results)
      min_iterations = results[w_best]
      if abs(min_iterations_old - min_iterations) < 2:
        print(f"网格划分{_}次，w已收敛")
        break
      else:
        min_iterations_old = min_iterations
        print(min_iterations)
      left = max(1.0, w_best - step)
      right = min(2.0, w_best + step)
      step=step/5
      omega_values = np.arange(left, right, step)
  
    best_omega = min(results, key=results.get)
    T_final = T.copy()
    SOR(T_final, best_omega, beta, max_iter, tolerance)
    print(f"Nx={Nx},Ny={Ny},Best ω = {best_omega}, iterations = {results[best_omega]}")
    draw(Lx,Ly,Nx,Ny,T_final,best_omega)

    w_list, iter_list = zip(*sorted(results.items()))

    plt.figure(figsize=(8, 4))
    plt.plot(w_list, iter_list, marker='o', markersize=3, label='SOR 迭代次数')
    plt.axvline(1.90752, color='red', linestyle='--', label='最优 ω ≈ 1.9075')
    plt.xlabel("松弛因子 ω")
    plt.ylabel("迭代次数")
    plt.title("SOR收敛速度随松弛因子的变化")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return best_omega
#调用
#修改网格划分

grid_sizes = []
omega_values = []
for i in range(1,2):
    Nx=75
    Ny=60
    omega=maincode(Nx,Ny) # 网格划分
    total_grid = Nx * Ny
    grid_sizes.append(total_grid)
    omega_values.append(omega)


# 设置支持中文的字体（需根据系统实际字体名称调整）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统黑体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 5), dpi=100)
plt.plot(grid_sizes, omega_values, 
         marker='o', 
         markersize=8,
         linestyle='--',
         color='#2c7bb6',
         linewidth=2,
         markerfacecolor='#d7191c',
         markeredgecolor='black')

# 图表装饰
plt.title("Ω 值与网格规模的关系", fontsize=14, pad=20)
plt.xlabel("总网格数 (Nx × Ny)", fontsize=12)
plt.ylabel("Ω 值", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 显示图表
plt.show()
'''
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
'''