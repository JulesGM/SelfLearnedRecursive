from pathlib import Path
import pickle

SCRIPT_DIR = Path(__file__).absolute().parent
DATA_DIR = SCRIPT_DIR / "data"


for path in DATA_DIR.glob("*.json.pkl"):
    print(f"Doing {path}")
    with open(path, "rb") as f:
        dataset = pickle.load(f)
        config = dataset["condig"]
        del dataset["condig"]
        dataset["config"] = config
    
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
