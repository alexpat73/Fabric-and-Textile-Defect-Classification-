import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    MODEL_DIR, OUTPUT_DIR, MODELS,
)
from dataset import get_dataloaders
from models import get_model
from utils import save_class_names, get_device


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt     = torch.exp(-ce)
        focal  = (1 - pt) ** self.gamma * ce
        return focal.mean()


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, macro_f1


def train_model(model_name, train_loader, val_loader, num_classes,
                class_weights, device):
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = os.path.join(MODEL_DIR, f'{model_name}_best.pth')

    model     = get_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    history      = {'train_loss': [], 'val_loss': [],
                    'train_acc': [],  'val_acc': [],
                    'train_f1': [],   'val_f1': []}
    best_val_f1  = -1.0
    best_epoch   = 0
    patience_ctr = 0

    print(f'\n{"="*55}')
    print(f'  Training: {model_name.upper()}  |  device: {device}')
    print(f'{"="*55}')

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_acc, vl_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, False)

        scheduler.step(vl_f1)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(vl_f1)

        print(f'Epoch {epoch:>3}/{NUM_EPOCHS} '
              f'| train loss {tr_loss:.4f} acc {tr_acc:.3f} F1 {tr_f1:.3f} '
              f'| val loss {vl_loss:.4f} acc {vl_acc:.3f} F1 {vl_f1:.3f} '
              f'| {time.time()-t0:.1f}s')

        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), checkpoint)
            print(f'  ✓ Best val F1 = {best_val_f1:.4f} (saved)')
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f'  Early stopping at epoch {epoch}.')
                break

    print(f'\n  Best val F1 = {best_val_f1:.4f} at epoch {best_epoch}')
    return {'history': history, 'best_epoch': best_epoch,
            'best_val_f1': best_val_f1, 'checkpoint': checkpoint}


if __name__ == '__main__':
    device = get_device()
    train_loader, val_loader, test_loader, class_names, class_weights = \
        get_dataloaders()
    num_classes = len(class_names)
    save_class_names(class_names)

    all_results = {}
    for model_name in MODELS:
        result = train_model(model_name, train_loader, val_loader,
                             num_classes, class_weights, device)
        all_results[model_name] = result

        # Save after each model in case of interruption
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = {k: {kk: vv for kk, vv in v.items() if kk != 'checkpoint'}
               for k, v in all_results.items()}
        with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
            json.dump(out, f, indent=2)
        print(f'  Results saved after {model_name}')

    print('\n✓ Training complete. Results saved to outputs/training_results.json')