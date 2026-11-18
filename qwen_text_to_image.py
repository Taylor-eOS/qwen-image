import batch
import utils

default_res = 640
width_input = input(f"Enter width ({default_res}): ").strip()
WIDTH = int(width_input) if width_input else default_res
height_input = input(f"Enter height ({WIDTH}): ").strip()
HEIGHT = int(height_input) if height_input else WIDTH

def run_generation(prompt):
    seed = utils.choose_seed()
    output = utils.make_output("text_to_image", seed)
    extra = [
        "-W", str(WIDTH),
        "-H", str(HEIGHT),
        "--seed", str(seed),
        "-o", str(output),
    ]
    return utils.run_sd(utils.build_command(extra, prompt), output)

def main():
    utils.check_size(WIDTH, HEIGHT)
    prompts = batch.load_prompts()
    utils.ensure_sd_cli()
    utils.fetch_models(vision=False)
    batch.run_all(prompts, run_generation)

if __name__ == "__main__":
    main()
