import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
# BASE_DIR resolves to wherever this file lives — works both locally
# (your Mac) and on Hugging Face Spaces (/home/user/app/).
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = BASE_DIR
OUTPUT_DIR = BASE_DIR
MODEL_DIR  = BASE_DIR

# ─── Dataset ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
RANDOM_SEED  = 42

# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7

# ─── Models ────────────────────────────────────────────────────────────────────
MODELS = ['custom_cnn', 'efficientnet_v2_s', 'convnext_tiny']

# ─── Explainability ────────────────────────────────────────────────────────────
NUM_EXPLAIN_SAMPLES = 10

# ─── Augmentation ──────────────────────────────────────────────────────────────
USE_AUGMENTATION = True
