import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from torchvision import transforms
import json

from config import MODEL_DIR, OUTPUT_DIR, BASE_DIR
from models import get_model
from explainability import GradCAM, get_target_layer, overlay

# ─── Class names ───────────────────────────────────────────────────────────────

_class_path = os.path.join(BASE_DIR, 'class_names.json')
with open(_class_path) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

# ─── Device ────────────────────────────────────────────────────────────────────
# Hugging Face free-tier Spaces run on CPU; CUDA is available on GPU Spaces.
# MPS is Mac-only and won't be present on Spaces — the fallback handles it.

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps')  if getattr(torch.backends, 'mps', None)
                             and torch.backends.mps.is_available() else
    torch.device('cpu')
)

# ─── Preprocessing ─────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Model cache ───────────────────────────────────────────────────────────────

_cache: dict = {}

def load_model(model_name: str):
    if model_name in _cache:
        return _cache[model_name]

    checkpoint = os.path.join(MODEL_DIR, f'{model_name}_best.pth')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Make sure the models/ folder contains the .pth files and was "
            "pushed to the Space with Git LFS."
        )

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE)
    # Handle checkpoints saved as {"model_state": ...} dicts or raw state dicts
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.eval()
    _cache[model_name] = model
    return model


# ─── Core prediction function ──────────────────────────────────────────────────

def predict(pil_image, model_name: str):
    if pil_image is None:
        return {}, None, "⚠️ Please upload a fabric image first."

    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        return {}, None, f"❌ {e}"

    # ── Forward pass (no grad needed for probabilities)
    img_t = TRANSFORM(pil_image.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_t)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # ── Grad-CAM  (needs a fresh tensor with grad enabled)
    img_t2 = TRANSFORM(pil_image.convert('RGB')).unsqueeze(0).to(DEVICE)
    img_t2.requires_grad_(True)

    target_layer = get_target_layer(model, model_name)
    gcam         = GradCAM(model, target_layer)
    with torch.enable_grad():
        hmap, _ = gcam.generate(img_t2, class_idx=pred_idx)
    gcam.remove()

    # ── Build heatmap overlay
    img_np    = np.array(pil_image.convert('RGB').resize((224, 224)))
    blended   = (overlay(img_np, hmap, alpha=0.5) * 255).astype(np.uint8)
    gradcam_img = Image.fromarray(blended)

    # ── Top-5 label dict for gr.Label
    top5_idx   = np.argsort(probs)[::-1][:5]
    label_dict = {CLASS_NAMES[i]: float(probs[i]) for i in top5_idx}

    # ── Defect / clean banner
    is_defective = 'defective' in pred_class
    status_icon  = '🔴 DEFECT DETECTED' if is_defective else '🟢 CLEAN'
    caption = (
        f"**{status_icon}**\n\n"
        f"**Class:** {pred_class.replace('_', ' ').title()}  \n"
        f"**Confidence:** {confidence:.1%}  \n"
        f"**Model:** {model_name.replace('_', ' ').upper()}"
    )

    return label_dict, gradcam_img, caption


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

def build_app():
    model_choices = ['efficientnet_v2_s', 'custom_cnn']

    with gr.Blocks(
        title='Textile Defect Inspector',
        theme=gr.themes.Soft(),
        css="""
        .title  { text-align: center; }
        .footer { text-align: center; font-size: 0.8rem; color: #888; margin-top: 1rem; }
        """
    ) as demo:

        gr.Markdown(
            """
            <div class="title">
            <h1> Textile Defect Inspector</h1>
            <p>Upload a fabric image to classify its type and condition,
            and see a <b>Grad-CAM heatmap</b> showing which regions
            influenced the model's decision.</p>
            </div>
            """,
        )

        with gr.Row():
            # ── Left column: inputs
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type='pil',
                    label='Fabric Image',
                    image_mode='RGB',
                )
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value='efficientnet_v2_s',
                    label='Model',
                    info='EfficientNetV2-S uses transfer learning; '
                         'Custom CNN is trained from scratch.',
                )
                run_btn = gr.Button('Analyse ▶', variant='primary')

                gr.Examples(
                    examples=["001-018.png","001-033.png","001-034.png","001-042.png","001-056.png","001-065.png","001-137.png","001-148.png","001-154.png"],   # add relative paths to sample images here
                    inputs=img_input,
                    label='Example images',
                )

            # ── Right column: outputs
            with gr.Column(scale=1):
                caption_out  = gr.Markdown(label='Result')
                gradcam_out  = gr.Image(
                    label='Grad-CAM — where the model looked',
                    show_label=True,
                )
                label_out = gr.Label(
                    label='Top 5 class probabilities',
                    num_top_classes=5,
                )

        run_btn.click(
            fn=predict,
            inputs=[img_input, model_dropdown],
            outputs=[label_out, gradcam_out, caption_out],
        )

        # Also run on image upload for faster UX
        img_input.change(
            fn=predict,
            inputs=[img_input, model_dropdown],
            outputs=[label_out, gradcam_out, caption_out],
        )

        gr.Markdown(
            """
            <div class="footer">
            Explainable Transfer Learning for Fabric and Textile Defect Classification ·
            Ten Fabrics Dataset · EfficientNetV2-S vs Custom CNN · PyTorch
            </div>
            """,
        )

    return demo


if __name__ == '__main__':
    app = build_app()
    app.launch()   # no share=True needed — Spaces handles the public URL
