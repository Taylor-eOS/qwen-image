import os
import sys
import time
import json
import shutil
import subprocess
import http.client
import urllib.error
import urllib.request
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is missing. Run: sudo apt install python3-pil")

WORK_DIR = Path.home() / "qwen_image_edit"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "image.jpg"
LONG_SIDE = 512
SEED = 42
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
SRC_DIR = WORK_DIR / "stable-diffusion.cpp"
BUILD_DIR = SRC_DIR / "build"
MODELS_DIR = WORK_DIR / "models"
OUT_DIR = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "qwen-image-edit-test"}
FILES = {}
SD_CLI = None
INPUT_IMAGE = None
PROMPT = ""
OUTPUT = None

def read_inputs():
    global INPUT_IMAGE, PROMPT, OUTPUT
    answer = input(f"Path to the image to edit [{DEFAULT_IMAGE}]: ").strip().strip("'\"")
    INPUT_IMAGE = Path(answer or DEFAULT_IMAGE).expanduser()
    if not INPUT_IMAGE.is_file():
        sys.exit(f"Image not found: {INPUT_IMAGE}")
    PROMPT = input("Edit instruction: ").strip()
    if not PROMPT:
        sys.exit("The instruction is empty")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT = OUT_DIR / f"{INPUT_IMAGE.stem}_edit_{time.strftime('%Y%m%d_%H%M%S')}.png"

def find_sd_cli():
    return next((path for path in BUILD_DIR.rglob("sd-cli") if path.is_file()), None)

def ensure_sd_cli():
    global SD_CLI
    SD_CLI = find_sd_cli()
    if SD_CLI:
        return
    missing = [tool for tool in ("git", "cmake", "make", "c++") if shutil.which(tool) is None]
    if missing:
        sys.exit(f"Missing build tools: {', '.join(missing)}. Run: sudo apt install git cmake build-essential")
    if not SRC_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", SD_CPP_URL, str(SRC_DIR)], check=True)
    subprocess.run(["cmake", "-S", str(SRC_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "-j", str(os.cpu_count() or 2)], check=True)
    SD_CLI = find_sd_cli()
    if not SD_CLI:
        sys.exit("The build finished but sd-cli was not found")

def list_repo_files(repo):
    request = urllib.request.Request(f"{HF_URL}/api/models/{repo}", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return [item["rfilename"] for item in json.load(response)["siblings"]]

def resolve_name(repo, suffix, prefix=""):
    names = list_repo_files(repo)
    matches = [name for name in names if Path(name).name.startswith(prefix) and name.endswith(suffix)]
    if len(matches) != 1:
        sys.exit(f"{repo}: expected one file starting with '{prefix}' and ending with '{suffix}', found {matches}. Files in the repo: {names}")
    return matches[0]

def download(repo, name):
    destination = MODELS_DIR / repo.replace("/", "--") / Path(name).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return str(destination)
    partial = destination.with_name(destination.name + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    headers = dict(HEADERS)
    if done:
        headers["Range"] = f"bytes={done}-"
    request = urllib.request.Request(f"{HF_URL}/{repo}/resolve/main/{name}", headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416 and done:
            partial.rename(destination)
            return str(destination)
        raise
    resumed = response.status == 206
    written = done if resumed else 0
    total = written + int(response.headers.get("Content-Length", 0))
    with response, open(partial, "ab" if resumed else "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            print(f"\r{destination.name}: {written / 1e9:.2f} of {total / 1e9:.2f} GB", end="", flush=True)
    print()
    if total and written != total:
        sys.exit(f"{destination.name} is incomplete, run the script again to resume")
    partial.rename(destination)
    return str(destination)

def fetch_models():
    FILES["denoiser"] = download(DENOISER_REPO, resolve_name(DENOISER_REPO, DENOISER_SUFFIX))
    FILES["encoder"] = download(ENCODER_REPO, resolve_name(ENCODER_REPO, ENCODER_SUFFIX))
    FILES["mmproj"] = download(ENCODER_REPO, resolve_name(ENCODER_REPO, MMPROJ_SUFFIX, MMPROJ_PREFIX))
    FILES["vae"] = download(VAE_REPO, VAE_FILE)
    if USE_UNCENSORED_LORA:
        FILES["lora"] = download(LORA_REPO, LORA_FILE)

def target_size():
    with Image.open(INPUT_IMAGE) as image:
        width, height = image.size
    scale = LONG_SIDE / max(width, height)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)

def build_command():
    width, height = target_size()
    prompt = PROMPT
    command = [
        str(SD_CLI),
        "--diffusion-model", FILES["denoiser"],
        "--vae", FILES["vae"],
        "--llm", FILES["encoder"],
        "--llm_vision", FILES["mmproj"],
        "-r", str(INPUT_IMAGE),
        "--sigmas", "qwen-image-2.1-turbo",
        "--steps", "8",
        "--cfg-scale", "1.0",
        "--sampling-method", "euler",
        "-W", str(width),
        "-H", str(height),
        "--seed", str(SEED),
        "-o", str(OUTPUT),
        "-v",
    ]
    if "lora" in FILES:
        prompt = f"<lora:{Path(FILES['lora']).stem}:1> {prompt}"
        command += ["--lora-model-dir", str(Path(FILES["lora"]).parent)]
    return command + ["-p", prompt]

def run_edit():
    started = time.time()
    result = subprocess.run(build_command())
    if result.returncode != 0:
        sys.exit(f"sd-cli failed with exit code {result.returncode}. Code -9 means the system ran out of memory")
    print(f"Saved {OUTPUT} in {(time.time() - started) / 60:.1f} minutes")

def main():
    read_inputs()
    ensure_sd_cli()
    try:
        fetch_models()
    except (OSError, http.client.HTTPException) as error:
        sys.exit(f"Download interrupted: {error}. Run the script again to resume")
    run_edit()

if __name__ == "__main__":
    main()
