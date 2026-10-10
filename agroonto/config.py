"""
AgroOnto-VLM configuration.
All hyperparameters and dataset paths in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_MODEL = "ViT-B/32"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_BLOCKS = 6

EPOCHS = 30
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42

LR_LORA = 1e-4
LR_HEAD = 1e-3
LR_DELTA = 1e-4
LR_LOGIT_SCALE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

LAMBDA_UNIFORM = 0.10
LAMBDA_PROTO = 0.0

PLANTVILLAGE_DIR = os.path.join(BASE_DIR, "data", "plantvillage")
DEEPWEEDS_IMG_DIR = os.path.join(BASE_DIR, "data", "deepweeds", "images")
DEEPWEEDS_LABEL_DIR = os.path.join(BASE_DIR, "data", "deepweeds", "labels")
IP102_IMG_DIR = os.path.join(BASE_DIR, "data", "ip102", "images")
IP102_LABEL_DIR = os.path.join(BASE_DIR, "data", "ip102", "labels")
