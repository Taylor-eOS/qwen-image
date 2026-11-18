import sys
from pathlib import Path
import utils
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is missing. Run: sudo apt install python3-pil")

IMAGE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_IMAGE = None
PROMPT = ""
WIDTH = 0
HEIGHT = 0
SEED = 0
OUTPUT = None

def target_size():
    with Image.open(INPUT_IMAGE) as image:
        width, height = image.size
    utils.check_size(width, height)
    return width, height

def find_images():
    found = [path for path in IMAGE_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(found, key=lambda path: path.name.lower())

def choose_image():
    images = find_images()
    if images:
        for number, path in enumerate(images, 1):
            print(f"{number}. {path.name}")
        answer = input("Image number or path [1]: ")
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
    return Path(answer).expanduser()

def read_inputs():
    global INPUT_IMAGE, PROMPT, WIDTH, HEIGHT, SEED, OUTPUT
    INPUT_IMAGE = choose_image()
    if not INPUT_IMAGE.is_file():
        sys.exit(f"Image not found: {INPUT_IMAGE}")
    PROMPT = input("Edit instruction: ").strip()
    if not PROMPT:
        sys.exit("The instruction is empty")
    WIDTH, HEIGHT = target_size()
    SEED = utils.choose_seed()
    OUTPUT = utils.make_output(f"{INPUT_IMAGE.stem}_edit", SEED)

def build_command():
    extra = [
        "-r", str(INPUT_IMAGE),
        "-W", str(WIDTH),
        "-H", str(HEIGHT),
        "--seed", str(SEED),
        "-o", str(OUTPUT),
    ]
    return utils.build_command(extra, PROMPT)

def run_edit():
    utils.run_sd(build_command(), OUTPUT)

def main():
    read_inputs()
    utils.ensure_sd_cli()
    utils.fetch_models(vision=True)
    run_edit()

if __name__ == "__main__":
    main()
