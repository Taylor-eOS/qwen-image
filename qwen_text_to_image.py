import sys
import utils

WIDTH = 1024
HEIGHT = 1024
PROMPT = ""
SEED = 0
OUTPUT = None

def read_inputs():
    global PROMPT, SEED, OUTPUT
    utils.check_size(WIDTH, HEIGHT)
    PROMPT = input("Image description: ").strip()
    if not PROMPT:
        sys.exit("The description is empty")
    SEED = utils.choose_seed()
    OUTPUT = utils.make_output("text_to_image", SEED)

def build_command():
    extra = [
        "-W", str(WIDTH),
        "-H", str(HEIGHT),
        "--seed", str(SEED),
        "-o", str(OUTPUT),
    ]
    return utils.build_command(extra, PROMPT)

def run_generation():
    utils.run_sd(build_command(), OUTPUT)

def main():
    read_inputs()
    utils.ensure_sd_cli()
    utils.fetch_models(vision=False)
    run_generation()

if __name__ == "__main__":
    main()
