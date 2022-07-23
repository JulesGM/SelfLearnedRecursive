import rich
import time
import os
import subprocess
import sys
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).absolute().parent

def cmd(command):
    output = subprocess.check_output(command).strip().decode().split("\n")
    return [x for x in output if x]

def start(command):
    return subprocess.Popen(command)


def main():
    find_command = ["find", str(SCRIPT_DIR / "log_results/oracle"), str(SCRIPT_DIR / "log_results/basic"), "-iname", "*ckpt"]
    launch_command = ["python", "bin_main_launcher.py", "resume"]

    finds = cmd(find_command)
    if not finds:
        rich.print("Didn't find anything with command:\n")
        rich.print(find_command)
        return
    # rich.print(finds)    

    targets = []
    for line in finds:
        epoch = int(re.match(r".*epoch=(\w+).*", line).group(1)); 
        if epoch <= 60:
            target = Path(line).parent.parent.parent.parent.parent / "specific_config.json"
            targets.append(target)

    rich.print(targets)
    procs = [start(launch_command + [target]) for target in targets]
    [proc.wait() for proc in procs]
    rich.print("Done.")    


if __name__ == "__main__":
    main()
