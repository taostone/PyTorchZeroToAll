import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import interp1d

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the layers of the network.
        Always call super().__init__() first!
        """
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def test_plt(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

    def test_power(self):
        x = torch.randn(5, requires_grad=True)
        y = x.pow(2)
        print(x.equal(y.grad_fn._saved_self))  # True
        print(x is y.grad_fn._saved_self)  # True
        
        # x = torch.tensor([2.0, 3.0, 4.0])
        # y = torch.pow(x, 3)
        # print(f"Input Tensor: {x}")
        # print(f"Tensor raised to power 3: {y}")
    
if __name__ == "__main__":
    print("--- PyTorch Model Debugging ---")
    
    # 1. Hyperparameters
    IN_SIZE, HIDDEN, CLASSES = 10, 20, 3
    
    # 2. Instantiate the model
    model = SimpleClassifier(IN_SIZE, HIDDEN, CLASSES)


    # begin for the test only
    model.test_power()
    # end for the test
    print(f"Model Structure:\n{model}")

    # 3. Create dummy data (Batch Size = 5, Input Features = 10)
    # This is the best way to debug: check if the shapes match!
    dummy_input = torch.randn(5, IN_SIZE)
    
    # 4. Perform a forward pass
    try:
        output = model(dummy_input)
        print(f"\nForward pass successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape} (Matches [Batch, Classes])")
    except Exception as e:
        print(f"Error during forward pass: {e}")


