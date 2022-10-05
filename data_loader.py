import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from datasets import bcic4_2a
from utils import plv_tensor, corr_tensor, normalize_adj_tensor, segment_tensor, transpose_tensor


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        if args.labels != 'all':
            self.label_selection()
        self.torch_form()

    def load_data(self):
        s = self.args.train_subject[0] + 1
        if self.args.phase == 'train':
            self.X = np.load(f"./data/S{s:02}_X_train.npy")
            self.y = np.load(f"./data/S{s:02}_y_train.npy")
        else:
            self.X = np.load(f"./data/S{s:02}_X_test.npy")
            self.y = np.load(f"./data/S{s:02}_y_test.npy")


    def label_selection(self):
        idx = [0]
        for label in self.args.labels:
            if len(idx) == 1:
                idx = (self.y == label)
            else:
                idx += (self.y == label)
        self.X = self.X[idx]
        self.y = self.y[idx]
        for mapping, label in enumerate(np.unique(self.y)):
            self.y[self.y == label] = mapping



    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        self.X = self.X.unsqueeze(1)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample



def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"
    trainset = CustomDataset(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader
