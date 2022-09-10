# 확인필요

class UNet(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU())
        
        self.Conv2 = nn.Sequential(
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3),
        nn.ReLU())
        
        self.Conv3 = nn.Sequential(
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3),
        nn.ReLU())
        
        self.Conv4 = nn.Sequental(
        nn.Conv2d(256, 512, 3),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3),
        nn.ReLU())
        
        self.Conv5 = nn.Sequential(
        nn.Conv2d(512, 1024, 3),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, 3),
        nn.ReLU())
        
        self.ConvT1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        
        self.Conv6 = nn.Sequential(
        nn.Conv2d(1024, 512, 3),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3),
        nn.ReLU())
        
        self.ConvT2 = nn.ConvTranspose2d(512, 256, 2, 2)
        
        self.Conv7 = nn.Sequential(
        nn.Conv2d(512, 256, 3),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3),
        nn.ReLU())
        
        self.ConvT3 = nn.ConvTranspose2d(256, 128, 2, 2)
        
        self.Conv8 = nn.Sequential(
        nn.Conv2d(256, 128, 3),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3),
        nn.ReLU())
        
        self.ConvT4 = nn.ConvTranspose2d(128, 64, 2, 2)
        
        self.Conv9 = nn.Sequential(
        nn.Conv2d(128, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 2, 1))
        
        
    def forward(self, x):
        x0 = self.Conv1(x)
        
        x1 = self.pool(x0)
        x1 = self.Conv2(x1)
        
        x2 = self.pool(x1)
        x2 = self.Conv3(x2)
        
        x3 = self.pool(x2)
        x3 = self.Conv4(x3)
        
        x_ = self.pool(x3)
        x_ = self.Conv5(x_)
        x_ = self.ConvT1(x_)
        
        x_ = torch.cat((x_, x3), dim=1)
        
        x_ = self.Conv6(x_)
        x_ = self.ConvT2(x_)
        
        x_ = torch.cat((x_, x2), dim=1)
        
        x_ = self.Conv7(x_)
        x_ = self.ConvT3(x_)
        
        x_ = torch.cat((x_, x1), dim=1)
        
        x_ = self.Conv8(x_)
        x_ = self.ConvT4(x_)
        
        x_ = torch.cat((x_, x0), dim=1)
        
        x_ = self.Conv9(x_)
        
        return x_
