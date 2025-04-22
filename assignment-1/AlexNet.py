import torch
import torch.nn as nn

# differs from original AlexNet
# adding batch normalization between Conv and ReLU
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # 可改成自己的類別數
        super().__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # pool1
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            # conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),           
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # pool2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),          
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),          
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # pool5
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # flatten後輸入
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.init_weights()
        print("初始化完成")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)