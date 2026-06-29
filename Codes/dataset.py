import os
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    RANDOM_SEED, USE_AUGMENTATION,
)


def get_transforms(split):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == 'train' and USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class TextileDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(data_dir=DATA_DIR):
    # Discover classes and images
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_paths, all_labels, all_groups = [], [], []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() not in valid_exts:
                continue
            all_paths.append(os.path.join(cls_dir, fname))
            all_labels.append(class_to_idx[cls])
            # Group by first 6 characters of filename e.g. "001-04"
            all_groups.append(fname[:6])

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)

    print(f'[Dataset] {len(all_paths)} images | {len(class_names)} classes')

    # Group-aware stratified split — pass 1: carve out test set
    def split_once(paths, labels, groups, test_frac):
        n_splits = round(1 / test_frac)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=RANDOM_SEED)
        for train_idx, test_idx in sgkf.split(paths, labels, groups):
            return train_idx, test_idx

    remaining_idx, test_idx = split_once(
        all_paths, all_labels, all_groups, TEST_SPLIT
    )

    val_frac_of_remaining = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = split_once(
        all_paths[remaining_idx],
        all_labels[remaining_idx],
        all_groups[remaining_idx],
        val_frac_of_remaining,
    )
    train_idx = remaining_idx[train_idx]
    val_idx   = remaining_idx[val_idx]

    print(f'[Dataset] train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}')

    def make_samples(indices):
        return [(all_paths[i], all_labels[i]) for i in indices]

    train_ds = TextileDataset(make_samples(train_idx), get_transforms('train'))
    val_ds   = TextileDataset(make_samples(val_idx),   get_transforms('val'))
    test_ds  = TextileDataset(make_samples(test_idx),  get_transforms('test'))

    # Class weights for focal loss
    label_counts  = np.bincount(all_labels[train_idx], minlength=len(class_names))
    class_weights = torch.tensor(1.0 / (label_counts + 1e-6), dtype=torch.float32)
    class_weights /= class_weights.sum()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names, class_weights