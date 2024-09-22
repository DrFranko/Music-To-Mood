import torch.nn as nn
import torch

class EmotionRec(nn.Module):
    def __init__(self):
        super(EmotionRec, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,4)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x