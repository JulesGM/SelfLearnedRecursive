from pathlib import Path
import sys

import fire
import rich

import jstyleson

import pretty_traceback
pretty_traceback.install()

SCRIPT_DIR = Path(__file__).absolute().parent

class Config:
    def __init__(self):
        self.configs = self._parse_file()


    def list_configs(self):
        rich.print(f"{list(self.configs.keys())}")

    default = list_configs

    def help(self):
        self.list_configs()
        for name, config in self.configs.items():
            rich.print(f"\"[bold blue]{name}[/]\":")
            rich.print(config)

    @classmethod
    def _parse_file(cls):
        with open(SCRIPT_DIR / ".vscode" / "launch.json") as f:
            complete = jstyleson.load(f)["configurations"]
        
        configs = {}
        for entry in complete:
            assert entry["name"] not in configs
            configs[entry["name"]] = entry
        return configs

    def main(self, name=None):
        if name is None:
            rich.print("[bold red]No config name given[/]. Here are those that are available:")
            self.list_configs()
            return

        if name in self.configs:
            active = self.configs[name]
            rich.print(active)
            args  = active.get("args", [])
            type_ = active.get("type", None)
            request = active.get("request", None)
            program = active.get("program", None)
            assert type_ == "python", type_
            assert request == "launch", request
            assert program.startswith("${workspaceFolder}/"), program
            program = SCRIPT_DIR / program[len("${workspaceFolder}/"):]
            rich.print(program)

        else:
            rich.print(f"No such config \"{name}\".")
            self.list_configs()



if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(Config())
    else:
        fire.Fire(Config(), command="list_configs")        