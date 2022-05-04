import wandb
import pretty_traceback
pretty_traceback.install()

def main():
    # Get all runs
    runs = list(wandb.Api().runs())
    # Get all short runs
    print(vars(runs[0].summary))

if __name__ == "__main__":
    main()