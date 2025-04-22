import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerCNN(nn.Module):
    def __init__(self):
        super(TwoLayerCNN, self).__init__()
        # 第一個卷積層: 輸入通道數為1，輸出通道數為16，卷積核大小為3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 第二個卷積層: 輸入通道數為16，輸出通道數為32，卷積核大小為3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 全連接層
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 假設輸入影像大小為28x28
        self.fc2 = nn.Linear(128, 10)  # 假設有10個分類

    def forward(self, x):
        # 第一層卷積 + ReLU + 最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 第二層卷積 + ReLU + 最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 展平
        x = x.view(x.size(0), -1)
        # 全連接層
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 測試模型結構
if __name__ == "__main__":
    model = TwoLayerCNN()
    print(model)
    # 測試輸入
    sample_input = torch.randn(1, 1, 28, 28)  # Batch size = 1, 單通道, 28x28影像
    output = model(sample_input)
    print(output.shape)  # 預期輸出形狀: [1, 10]