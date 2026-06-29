import os
import json
import torch
from config import OUTPUT_DIR


def save_class_names(class_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'class_names.json')
    with open(path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f'[utils] Class names saved → {path}')


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def format_metric_table(report):
    header = '| Model | Acc | Bal-Acc | Macro-F1 | Wt-F1 | Infer(ms) | Size(MB) |'
    sep    = '|---|---|---|---|---|---|---|'
    rows   = [header, sep]
    for name, m in report.items():
        rows.append(
            f"| {name} "
            f"| {m['accuracy']:.4f} "
            f"| {m['balanced_accuracy']:.4f} "
            f"| {m['macro_f1']:.4f} "
            f"| {m['weighted_f1']:.4f} "
            f"| {m['inference_ms']:.1f} "
            f"| {m['size_mb']:.1f} |"
        )
    return '\n'.join(rows)