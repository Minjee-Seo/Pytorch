import torch
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 1

# 임의의 데이터셋 생성

x = [[0,0,0], [0,0,1], [0,1,0],
    [1,0,0], [0,1,1], [1,1,0],
    [1,0,1], [1,1,1]]

y = [0, 1, 2, 3, 4, 5, 6, 7]

x = torch.Tensor(x).float()
y = torch.Tensor(y).long()

# Dataset class 이용해 dataset 정의

class Data(Dataset):
    def __init__(self):
        self.x_data = x
        self.y_data = y
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
      
# 인스턴스 생성

dataset = Data()

loader = DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True)
