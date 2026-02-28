import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


def _build_train_transforms() -> transforms.Compose:
    """
    Strong geometric + occlusion augmentations for the source training set.
    Designed for single-channel 28×28 float tensors.

    Permitted transforms applied
    ----------------------------
    RandomCrop       – random spatial crop with zero-padding (4 px)
    RandomHorizontalFlip  – left-right mirror with p=0.5
    RandomAffine     – combined rotation (±15°) + translation (±10 %)
    ColorJitter      – brightness & contrast only (grayscale; no saturation/hue)
    RandomErasing    – Cutout-style occlusion square dropped at random
    """
    return transforms.Compose([
        transforms.RandomCrop(28, padding=4, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
    ])


def _build_val_transforms() -> transforms.Compose:
    """
    Mild geometric augmentations for the validation set.
    Keeps temperature-scaling and BBSE calibration stable while adding
    enough diversity to smooth the confusion matrix estimates.

    Permitted transforms applied
    ----------------------------
    RandomCrop       – minimal padding (2 px) for slight position jitter
    RandomHorizontalFlip  – left-right mirror with p=0.5
    ColorJitter      – very subtle brightness & contrast shift
    """
    return transforms.Compose([
        transforms.RandomCrop(28, padding=2, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class CustomPTDataset(Dataset):
    def __init__(
        self,
        path: str,
        return_index: bool = False,
        split: str = "val",
        augmentation_factor: int = 1,
    ):
        """
        Parameters
        ----------
        path                : path to a .pt file containing keys 'x' (images) and 'y' (labels)
        return_index        : if True, __getitem__ returns (x, y, idx) instead of (x, y)
        split               : one of 'train' | 'val' | 'target'
                              'train'  – strong augmentations (crop, flip, affine, jitter, erasing)
                              'val'    – mild augmentations   (crop, flip, subtle jitter)
                              'target' – no augmentation      (clean inference on test domain)
        augmentation_factor : virtual dataset multiplier (>= 1).
                              __len__ returns N * augmentation_factor; each logical index maps
                              back to the original sample via idx % N, so every original image
                              is visited augmentation_factor times per epoch with a fresh
                              random transform each time.
        """
        data = torch.load(path, weights_only=True)

        self.images = data["x"].float()
        self.return_index = return_index
        self.labels = data["y"].long()
        self.augmentation_factor = max(1, augmentation_factor)
        self._base_len = len(self.images)

        if split == "train":
            self.transform = _build_train_transforms()
        elif split == "val":
            self.transform = _build_val_transforms()
        else:  # "target" or any other value → no augmentation
            self.transform = transforms.Compose([])

    def __len__(self) -> int:
        return self._base_len * self.augmentation_factor

    def __getitem__(self, idx: int):
        orig_idx = idx % self._base_len
        x = self.transform(self.images[orig_idx])
        y = self.labels[orig_idx]

        if self.return_index:
            return x, y, orig_idx

        return x, y
