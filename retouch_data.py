import argparse
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
SCRIPT_DIR = Path(__file__).absolute().parent
DATA_DIR = SCRIPT_DIR / "data"

def main():
    
    files = list(Path(DATA_DIR).glob("*.pkl"))

    for file in tqdm(files):
        with open(file, "rb") as f:
            obj = pickle.load(f)
        print("Loaded data")
        print(obj["config"])
        config = argparse.Namespace(**obj["config"])
        obj["config"]["output_name"] = f"{config.max_total_length}_{config.max_answer_length}_{config.max_depth}_{config.max_qty_per_level}.json"

        print("Saving data")
        with open(file, "wb") as f:
            pickle.dump(obj, f)
        print("Saved data")

if __name__ == "__main__":
    main()
