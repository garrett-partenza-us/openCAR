import torch.nn as nn

class SupNet(nn.Module):
    def __init__(self):
        super(SupNet, self).__init__()
        
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=(5,5), stride=1, padding=2)
        self.cnn2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)
        self.cnn3 = nn.Conv2d(64, 12, kernel_size=(3,3), stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        
        
    def forward(self, x):
        x = nn.Tanh()(self.cnn1(x))
        x = nn.Tanh()(self.cnn2(x))
        x = self.cnn3(x)
        x = self.shuffle(x)
        return x.permute(0,2,3,1)