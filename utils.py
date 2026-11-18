import os
import sys
import json
import time
import shutil
import random
import subprocess
import http.client
import urllib.error
import urllib.request
from pathlib import Path
from settings import SD_CPP_URL, HF_URL, DENOISER_REPO, DENOISER_SUFFIX, ENCODER_REPO, ENCODER_SUFFIX, MMPROJ_PREFIX, MMPROJ_SUFFIX, VAE_REPO, VAE_FILE, LORA_REPO, LORA_FILE, SRC_DIR, BUILD_DIR, MODELS_DIR, OUT_DIR, HEADERS, ALIGN, MAX_PIXELS, SEED, USE_UNCENSORED_LORA

FILES = {}
SD_CLI = None

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

def find_local_names(repo, suffix, prefix):
    folder = MODELS_DIR / repo.replace("/", "--")
    if not folder.is_dir():
        return []
    return [path.name for path in folder.iterdir() if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)]

def resolve_name(repo, suffix, prefix=""):
    local = find_local_names(repo, suffix, prefix)
    if len(local) == 1:
        return local[0]
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

def fetch_models(vision):
    try:
        FILES["denoiser"] = download(DENOISER_REPO, resolve_name(DENOISER_REPO, DENOISER_SUFFIX))
        FILES["encoder"] = download(ENCODER_REPO, resolve_name(ENCODER_REPO, ENCODER_SUFFIX))
        if vision:
            FILES["mmproj"] = download(ENCODER_REPO, resolve_name(ENCODER_REPO, MMPROJ_SUFFIX, MMPROJ_PREFIX))
        FILES["vae"] = download(VAE_REPO, VAE_FILE)
        if USE_UNCENSORED_LORA:
            FILES["lora"] = download(LORA_REPO, LORA_FILE)
    except (OSError, http.client.HTTPException) as error:
        sys.exit(f"Download interrupted: {error}. Run the script again to resume")

def check_size(width, height):
    if width <= 0 or height <= 0 or width % ALIGN or height % ALIGN:
        sys.exit(f"The size {width}x{height} is not allowed, both sides must be positive multiples of {ALIGN}")
    if width * height > MAX_PIXELS:
        sys.exit(f"The size {width}x{height} has more than {MAX_PIXELS} pixels, raise MAX_PIXELS in settings.py if you want this")

def choose_seed():
    return SEED if SEED >= 0 else random.randrange(2 ** 31)

def make_output(stem, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{stem}_{seed}.png"

def build_command(extra, prompt):
    command = [
        str(SD_CLI),
        "--diffusion-model", FILES["denoiser"],
        "--vae", FILES["vae"],
        "--llm", FILES["encoder"],
        "--sigmas", "qwen-image-2.1-turbo",
        "--steps", "8",
        "--cfg-scale", "1.0",
        "--sampling-method", "euler",
        "-v",
    ]
    if "mmproj" in FILES:
        command += ["--llm_vision", FILES["mmproj"]]
    if "lora" in FILES:
        prompt = f"<lora:{Path(FILES['lora']).stem}:1> {prompt}"
        command += ["--lora-model-dir", str(Path(FILES["lora"]).parent)]
    return command + extra + ["-p", prompt]

def run_sd(command, output):
    started = time.time()
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"sd-cli failed with exit code {result.returncode}. Code -9 means the system ran out of memory")
        return False
    print(f"Saved {output} in {(time.time() - started) / 60:.1f} minutes")
    return True
