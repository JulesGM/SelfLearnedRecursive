"""
Deletes runs that are under a certain duration or that don't have a recorded duration at all.
"""
import rich
import wandb
import pretty_traceback
pretty_traceback.install()

NUM_MINUTES_TO_KEEP = 30
PROJECT = "julesgm/self_learned_explanations"

def parse_time(seconds):
    hours = seconds // 60 // 60
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    return hours, minutes, seconds

def main():
    assert NUM_MINUTES_TO_KEEP < 60, "Use hours if you want more than 59 minutes."
    # Get all runs
    runs = list(wandb.Api().runs(PROJECT))
    # Get all short runs
    for run in runs:
        print(run.name)
        print(run.state)
        if "_runtime" in run.summary:
            hours, minutes, seconds = parse_time(run.summary["_runtime"])
            print(f"Runtime: {hours}h {minutes}m {seconds}s")
            if run.state != "running" and minutes < NUM_MINUTES_TO_KEEP and hours == 0:
                run.delete()
                rich.print("[red]deleting")
        else:
            run.delete()
            print("No runtime.")
            rich.print("[red]deleting")
            print(run.summary)
        if "_step" in run.summary:
            print(run.summary["_step"])
        else:
            print("No step")
        print()


if __name__ == "__main__":
    main()
