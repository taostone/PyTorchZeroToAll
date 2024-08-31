import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 给定的新数据点
x_data_new = np.array([0, 1, 2, 4])
y_data_new = np.array([11, 6, -1, 3])

# 使用样条插值拟合新数据
f_spline = interp1d(x_data_new, y_data_new, kind='cubic')

# 生成更多点以绘制平滑的曲线
x_interp_new = np.linspace(min(x_data_new), max(x_data_new), 100)
y_interp_new = f_spline(x_interp_new)

# 绘制原始数据点和样条插值曲线
plt.figure(figsize=(8, 6))
plt.plot(x_data_new, y_data_new, 'ro', label='原始数据')
plt.plot(x_interp_new, y_interp_new, 'b-', label='样条插值曲线')
plt.title('函数的样条插值图像（新数据）')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
