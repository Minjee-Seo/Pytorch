class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.Conv1 = nn.Sequential( # -> a
        nn.Conv2d(1, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU())
        
        self.Conv2 = nn.Sequential( # -> b
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3),
        nn.ReLU())
        
        self.Conv3 = nn.Sequential( # -> c
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3),
        nn.ReLU())
        
        self.Conv4 = nn.Sequential( # -> d
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
        nn.Conv2d(64, 1, 1),
        nn.Sigmoid())
        
    def crop(self, x, output_size):
        transform = CenterCrop(output_size)
        return transform(x)
        
    def forward(self, x):
        a = self.Conv1(x)
        
        b = self.pool(a)
        b = self.Conv2(b)
        
        c = self.pool(b)
        c = self.Conv3(c)
        
        d = self.pool(c)
        d = self.Conv4(d)
        
        x_ = self.pool(d)
        x_ = self.Conv5(x_)
        x_ = self.ConvT1(x_)
        
        d = self.crop(d, x_.size(dim=2))
        x_ = torch.cat((x_, d), dim=1)
        
        x_ = self.Conv6(x_)
        x_ = self.ConvT2(x_)
        
        c = self.crop(c, x_.size(dim=2))
        x_ = torch.cat((x_, c), dim=1)
        
        x_ = self.Conv7(x_)
        x_ = self.ConvT3(x_)
        
        b = self.crop(b, x_.size(dim=2))
        x_ = torch.cat((x_, b), dim=1)
        
        x_ = self.Conv8(x_)
        x_ = self.ConvT4(x_)
        
        a = self.crop(a, x_.size(dim=2))
        x_ = torch.cat((x_, a), dim=1)
        
        x_ = self.Conv9(x_)
        
        return x_
