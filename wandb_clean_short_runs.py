import rich
import wandb
import pretty_traceback
pretty_traceback.install()

def main():
    # Get all runs
    runs = list(wandb.Api().runs("julesgm/self_learned_explanations"))
    # Get all short runs
    for run in runs:
        print(run.name)
        print(run.state)
        if "_runtime" in run.summary:
            hours = run.summary["_runtime"] // 60 // 60
            minutes = (run.summary["_runtime"] // 60) % 60
            seconds = run.summary["_runtime"] % 60
            print(f"Runtime: {hours}h {minutes}m {seconds}s")
            if run.state != "running" and minutes < 5 and hours == 0:
                rich.print(f"[red bold]Candidate.")
        else:
            print("No runtime")
            print(run.summary)
        if "_step" in run.summary:
            print(run.summary["_step"])
        else:
            print("No step")
        print()

if __name__ == "__main__":
    main()