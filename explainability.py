import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import MODEL_DIR, OUTPUT_DIR
from models import get_model


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, input, output):
        self.activations = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=img_tensor.shape[2:],
                                mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def get_target_layer(model, model_name):
    if model_name == 'custom_cnn':
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, torch.nn.Conv2d):
                return layer
    elif model_name in ('efficientnet_v2_s', 'convnext_tiny'):
        return model.features[-1]
    raise ValueError(f'Unknown model: {model_name}')


def overlay(img_np, heatmap, alpha=0.5):
    colormap  = cm.get_cmap('jet')
    heatmap_c = colormap(heatmap)[..., :3]
    blended   = alpha * heatmap_c + (1 - alpha) * img_np / 255.0
    return np.clip(blended, 0, 1)


def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    img  = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def save_gradcam_grid(samples, heatmaps, preds, class_names,
                      model_name, save_dir):
    n   = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (img_t, true_label, pred, hmap) in enumerate(
            zip(*zip(*samples), preds, heatmaps)):
        img_np  = denorm(img_t)
        blended = overlay(img_np, hmap)

        axes[i][0].imshow(img_np)
        axes[i][0].set_title(
            f'True: {class_names[true_label]}\nPred: {class_names[pred]}',
            fontsize=8)
        axes[i][0].axis('off')

        axes[i][1].imshow(blended)
        axes[i][1].set_title('Grad-CAM', fontsize=8)
        axes[i][1].axis('off')

    plt.suptitle(f'Grad-CAM — {model_name}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f'gradcam_{model_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grad-CAM saved → {path}')


# ─── Standalone script only ───────────────────────────────────────────────────

if __name__ == '__main__':
    from dataset import get_dataloaders

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    _, _, test_loader, class_names, _ = get_dataloaders()
    num_classes = len(class_names)

    save_dir = os.path.join(OUTPUT_DIR, 'explanations')
    os.makedirs(save_dir, exist_ok=True)

    for model_name in ['custom_cnn', 'efficientnet_v2_s']:
        print(f'\nGenerating Grad-CAM for {model_name}...')

        checkpoint = os.path.join(MODEL_DIR, f'{model_name}_best.pth')
        if not os.path.exists(checkpoint):
            print(f'  No checkpoint — skipping.')
            continue

        model = get_model(model_name, num_classes).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

        samples, heatmaps, preds = [], [], []
        target_layer = get_target_layer(model, model_name)
        gcam         = GradCAM(model, target_layer)

        defective_indices = [i for i, name in enumerate(class_names)
                             if 'defective' in name]

        for imgs, labels in test_loader:
            for i in range(len(imgs)):
                if len(samples) >= 5:
                    break
                if labels[i].item() not in defective_indices:
                    continue
                img_t = imgs[i:i + 1].to(device)
                img_t.requires_grad_(True)
                with torch.enable_grad():
                    hmap, pred = gcam.generate(img_t)
                samples.append((imgs[i], labels[i].item()))
                heatmaps.append(hmap)
                preds.append(pred)
            if len(samples) >= 5:
                break

        gcam.remove()
        save_gradcam_grid(samples, heatmaps, preds,
                          class_names, model_name, save_dir)

    print('\n✓ Grad-CAM complete. Images saved to outputs/explanations/')