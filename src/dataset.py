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
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
        is_labelled: bool = True,
        augmentation_factor: int = 1,
        indices: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        path                : path to a .pt file containing keys 'images' (and optionally 'labels')
        return_index        : if True, __getitem__ returns (x, y, idx) instead of (x, y)
        split               : one of 'train' | 'val' | 'test'
                              'train' – strong augmentations (crop, flip, affine, jitter, erasing)
                              'val'   – mild augmentations   (crop, flip, subtle jitter)
                              'test'  – no augmentation      (clean inference on test domain)
        is_labelled         : whether the file contains a 'labels' key
        augmentation_factor : virtual dataset multiplier (>= 1).
                              __len__ returns N * augmentation_factor; each logical index maps
                              back to the original sample via idx % N, so every original image
                              is visited augmentation_factor times per epoch with a fresh
                              random transform each time.
        indices             : optional list of integer indices to subset the dataset.
                              If provided, only those rows are kept (useful for train/val splits).
        """
        data = torch.load(path, weights_only=True)
        images = data["images"].float()

        if indices is not None:
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            images = images[idx_tensor]

        self.images = images
        self.is_labelled = is_labelled
        self.return_index = return_index

        if is_labelled:
            labels = data["labels"].long()
            if indices is not None:
                labels = labels[idx_tensor]
            self.labels = labels

        self.augmentation_factor = max(1, augmentation_factor)
        self._base_len = len(self.images)

        if split == "train":
            self.transform = _build_train_transforms()
        elif split == "val":
            self.transform = _build_val_transforms()
        else:  # "test" or any other value → no augmentation
            self.transform = transforms.Compose([torch.nn.Identity()])

    def __len__(self) -> int:
        return self._base_len * self.augmentation_factor

    def __getitem__(self, idx: int):
        orig_idx = idx % self._base_len
        x = self.transform(self.images[orig_idx])

        if self.is_labelled:
            y = self.labels[orig_idx]
            return (x, y, orig_idx) if self.return_index else (x, y)

        return (x, orig_idx) if self.return_index else x
