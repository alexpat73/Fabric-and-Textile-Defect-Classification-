import os
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from config import MODEL_DIR, OUTPUT_DIR, MODELS
from dataset import get_dataloaders
from models import get_model


def evaluate_on_test(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)


def measure_inference_time(model, device, n_runs=100):
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
    return (time.perf_counter() - t0) / n_runs * 1000


def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Confusion matrix saved → {path}')


def plot_training_curves(history, model_name, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['train_f1'], label='Train')
    axes[1].plot(epochs, history['val_f1'],   label='Val')
    axes[1].set_title(f'{model_name} — Macro-F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro-F1')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f'training_curves_{model_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Training curves saved  → {path}')


if __name__ == '__main__':
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, _, test_loader, class_names, _ = get_dataloaders()
    num_classes = len(class_names)

    with open(os.path.join(OUTPUT_DIR, 'training_results.json')) as f:
        training_results = json.load(f)

    report = {}

    for model_name in MODELS:
        print(f'\n{"─"*55}')
        print(f'  Evaluating: {model_name.upper()}')
        print(f'{"─"*55}')

        checkpoint = os.path.join(MODEL_DIR, f'{model_name}_best.pth')
        if not os.path.exists(checkpoint):
            print(f'  No checkpoint found — skipping.')
            continue

        model = get_model(model_name, num_classes).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        preds, labels = evaluate_on_test(model, test_loader, device)

        acc      = accuracy_score(labels, preds)
        bal_acc  = balanced_accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro',    zero_division=0)
        wt_f1    = f1_score(labels, preds, average='weighted', zero_division=0)
        cm       = confusion_matrix(labels, preds)
        infer_ms = measure_inference_time(model, device)

        total_params = sum(p.numel() for p in model.parameters())
        size_mb      = sum(p.nelement() * p.element_size()
                          for p in model.parameters()) / 1e6

        report[model_name] = {
            'accuracy'         : round(acc,      4),
            'balanced_accuracy': round(bal_acc,   4),
            'macro_f1'         : round(macro_f1,  4),
            'weighted_f1'      : round(wt_f1,     4),
            'inference_ms'     : round(infer_ms,  2),
            'parameters'       : total_params,
            'size_mb'          : round(size_mb,   2),
        }

        print(f'  Accuracy          : {acc:.4f}')
        print(f'  Balanced Accuracy : {bal_acc:.4f}')
        print(f'  Macro-F1          : {macro_f1:.4f}')
        print(f'  Weighted-F1       : {wt_f1:.4f}')
        print(f'  Inference time    : {infer_ms:.2f} ms')
        print(f'  Parameters        : {total_params:,}')
        print(f'  Model size        : {size_mb:.2f} MB')
        print(f'\n  Full per-class report:')
        print(classification_report(labels, preds,
                                    target_names=class_names, zero_division=0))

        plot_confusion_matrix(cm, class_names, model_name, OUTPUT_DIR)
        if model_name in training_results:
            plot_training_curves(training_results[model_name]['history'],
                                 model_name, OUTPUT_DIR)
        else:
            print(f'  No training history found for {model_name} — skipping curves.')

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n✓ Evaluation complete. Report saved to outputs/evaluation_report.json')