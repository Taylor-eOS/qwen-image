import sys
from pathlib import Path
import utils
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is missing. Run: sudo apt install python3-pil")

DEFAULT_IMAGE = Path(__file__).resolve().parent / "image.jpg"
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

def read_inputs():
    global INPUT_IMAGE, PROMPT, WIDTH, HEIGHT, SEED, OUTPUT
    answer = input(f"Path to the image to edit [{DEFAULT_IMAGE}]: ").strip().strip("'\"")
    INPUT_IMAGE = Path(answer or DEFAULT_IMAGE).expanduser()
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
