from pathlib import Path

from src.utils.common import DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, VALID_IMAGE_EXTENSIONS

OUTPUT_DIR = Path('outputs')
RAW_DIR = OUTPUT_DIR / 'raw'
HEATMAP_DIR = OUTPUT_DIR / 'heatmaps'
OVERLAY_DIR = OUTPUT_DIR / 'overlays'

DEFAULT_IMAGE_SIZE = IMAGE_SIZE
DEFAULT_DEVICE = DEVICE
DEFAULT_METHOD = 'gradcam'
AVAILABLE_METHODS = ['vanilla', 'gradcam']
AVAILABLE_MODELS = ['fusion_cam', 'attention', 'cam_head', 'baseline', 'maal', 'maal_g', 'maal_h']
DEFAULT_MODEL_TYPE = 'fusion_cam'
DEFAULT_TARGET_LAYER = 'encoder.layer4[-1].conv3'
DEFAULT_THRESHOLD = 0.5
