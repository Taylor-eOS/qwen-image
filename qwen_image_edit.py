import sys
from pathlib import Path
import batch
import utils
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is missing. Run: sudo apt install python3-pil")

IMAGE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def target_size(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    utils.check_size(width, height)
    return width, height

def find_images():
    found = [path for path in IMAGE_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "_seed" not in path.stem]
    return sorted(found, key=lambda path: path.name.lower())

def choose_image():
    images = find_images()
    if images:
        for number, path in enumerate(images, 1):
            print(f"{number}. {path.name}")
        answer = input("Image number or path (1): ")
    else:
        answer = input("No images found in the script folder. Path to the image: ")
    answer = answer.strip().strip("'\"")
    if images and not answer:
        return images[0]
    if answer.isdigit():
        index = int(answer)
        if not 1 <= index <= len(images):
            sys.exit(f"Invalid number: {index}")
        return images[index - 1]
    path = Path(answer).expanduser()
    if not path.is_file():
        sys.exit(f"Image not found: {path}")
    return path

def run_edit(prompt, image_path, width, height):
    seed = utils.choose_seed()
    output = utils.make_output(f"{image_path.stem}_edit", seed)
    extra = [
        "-r", str(image_path),
        "-W", str(width),
        "-H", str(height),
        "--seed", str(seed),
        "-o", str(output),
    ]
    return utils.run_sd(utils.build_command(extra, prompt), output)

def main():
    prompts = batch.load_prompts()
    image_path = choose_image()
    width, height = target_size(image_path)
    utils.ensure_sd_cli()
    utils.fetch_models(vision=True)
    batch.run_all(prompts, lambda prompt: run_edit(prompt, image_path, width, height))

if __name__ == "__main__":
    main()
