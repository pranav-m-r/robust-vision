import torch
from torch.utils.data import Dataset


class CustomPTDataset(Dataset):
    def __init__(self, path: str, return_index: bool = False):
        data = torch.load(path, weights_only=True)

        self.images = data["x"].float()
        self.return_index = return_index
        self.labels = data["y"].long()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        y = self.labels[idx]

        if self.return_index:
            return x, y, idx

        return x, y
