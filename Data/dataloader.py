from torch.utils.data import Dataset, DataLoader

loader = DataLoader(dataset = dataset, batch_size = 64, shuffle = True)
