from pathlib import Path

WORK_DIR = Path.home() / "qwen_image_edit"
SRC_DIR = WORK_DIR / "stable-diffusion.cpp"
BUILD_DIR = SRC_DIR / "build"
MODELS_DIR = WORK_DIR / "models"
OUT_DIR = Path(__file__).resolve().parent
ALIGN = 32
SEED = -1
MAX_PIXELS = 1024 * 1024
USE_UNCENSORED_LORA = False
SD_CPP_URL = "https://github.com/leejet/stable-diffusion.cpp"
HF_URL = "https://huggingface.co"
DENOISER_REPO = "AtomicChat/Qwen-Image-2.1-Turbo-GGUF"
DENOISER_SUFFIX = "AD-Q4_K.gguf"
ENCODER_REPO = "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"
ENCODER_SUFFIX = "AD-Q4_K.gguf"
MMPROJ_PREFIX = "mmproj"
MMPROJ_SUFFIX = "-F16.gguf"
VAE_REPO = "Comfy-Org/Qwen-Image-2.1"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"
LORA_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"
LORA_FILE = "qwen-image-2.1-uncensored-lora.safetensors"
HEADERS = {"User-Agent": "qwen-image-edit-test"}
