import pickle
import json
import datagen
import time

def main():
    # start = time.perf_counter()
    # with open("data/native.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(f"Loaded native in {time.perf_counter() - start:0.2f} seconds")

    start = time.perf_counter()
    with open("data/dicts.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"Loaded dicts from {time.perf_counter() - start:0.2f} seconds")

    start = time.perf_counter()
    with open("data/dataset.json", "r") as f:
        data = json.load(f)
    print(f"Loaded json from {time.perf_counter() - start:0.2f} seconds")
    
if __name__ == "__main__":
    main()