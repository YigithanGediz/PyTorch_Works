from torch.utils.data import Dataset
import torch
from torch import nn

class StarsDataset(Dataset):
    def __init__(self, x, y, transform = None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item], self.y[item]
        if self.transform:
            sample = self.transform(sample)

        sample = list(sample)
        sample[1] = sample[1].type(torch.long)

        return tuple(sample)

    def __len__(self):
        return self.n_samples

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.input_size = input_size
        self.classes = num_classes
        self.linear1 = nn.Linear(input_size, 16)
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.lrelu(out)
        out = self.linear2(out)
        out = self.lrelu(out)
        out = self.linear3(out)

        return out
