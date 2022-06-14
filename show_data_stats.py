import fire
from pathlib import Path
import pickle
import rich
SCRIPT_DIR = Path(__file__).absolute().parent
DATA_DIR = SCRIPT_DIR / "data"


def main(filename):
    data_path = DATA_DIR / Path(filename).name
    print(f"Loading the data from \"{data_path}\"")
    with open(data_path, "rb") as f:
        obj = pickle.load(f)
    
    print("Loaded the data.\n")
    rich.print("Config:\n{obj['config']}\n")
    train = {k: v["train"] for k, v in enumerate(obj["data"])}
    valid = {k: v["eval"] for k, v in enumerate(obj["data"])}
    train_lengths = {k: len(v) for k, v in train.items()}
    valid_lengths = {k: len(v) for k, v in valid.items()}
    rich.print(f"Train lengths:\n{train_lengths}\n")
    rich.print(f"Valid lengths:\n{valid_lengths}\n")
    

if __name__ == "__main__":
    fire.Fire(main)
