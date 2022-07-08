#!/usr/bin/env python3
"""
Assumes that this script is in the same directory as `.vsconde` folder.
It would make more sense as a package I guess. 

To run your config:
    python run.py run <config_name> <args>

To see the available config names:
    python run.py

To read the contents of a specific config: 
    python run.py help <config_name>
"""

import os
from pathlib import Path
import re
import sys

import fire  # type: ignore[import]
import jstyleson  # type: ignore[import]
import pretty_traceback  # type: ignore[import]
import rich

pretty_traceback.install()


SCRIPT_DIR = Path(__file__).absolute().parent
CONFIG_PATH = SCRIPT_DIR / ".vscode" / "launch.json"


def render_vscode_vars(vars_str):
    """Renders the VsCode config.json variable template.
    Uses ultra dumb string substitution to be as predictable
    as possible. Using a template renderer might have all sorts
    of other features that are not needed here & that may have
    unexpected side effects.
    """
    variables = dict(
        userHome=os.path.expanduser("~"),
        workspaceFolder=SCRIPT_DIR,
        workspaceFolderBasename=SCRIPT_DIR.name,
        cwd=os.getcwd(),
        pathSeparator=os.pathsep,
    )

    for name, value in variables.items():
        vars_str = vars_str.replace("${" + name + "}", str(value))

    # Make sure we parsed all the variables
    non_handled_matches = re.findall(r"\$\{.*\}", vars_str)
    assert not non_handled_matches, non_handled_matches
    return vars_str


class Config:
    def __init__(self):
        self._configs = self._parse_file()

    def _get_config(self, name):
        """Get a config by name or print a standard error message."""
        if name in self._configs:
            return self._configs[name]
        else:
            rich.print(
                f'[red]No such config "{name}".[/] '
                f"Available configs:\n{self.list_configs()}"
            )

    @classmethod
    def _parse_file(cls):
        """Parse the config file in it's default location.
        We assume that the script is in the same directory
        as the `.vscode` folder.
        """

        with CONFIG_PATH.open() as f:
            complete = jstyleson.load(f)["configurations"]

        configs = {}
        for entry in complete:
            assert entry["name"] not in configs
            configs[entry["name"]] = entry

        return configs

    def help(self, name=None):
        """Prints the named config or prints the name of the available configs."""
        if name is None:
            rich.print(
                "Type `python run.py help <config_name>` "
                "to get help on a specific config"
            )
            self.list_configs()
        else:
            rich.print(f'[bold blue]"{name}":[/]')
            rich.print(self._get_config(name))

    def list_configs(self):
        rich.print(f"{list(self._configs.keys())}")

    def run(self, name=None):
        if name is None:
            rich.print(
                "[bold red]No config name given[/]. "
                "Here are those that are available:"
            )
            self.list_configs()
            return

        active = self._get_config(name)
        args = active.get("args", [])
        type_ = active["type"]
        request = active["request"]
        program = active["program"]
        assert type_ == "python", type_
        assert request == "launch", request
        assert program.startswith("${workspaceFolder}/"), program

        program = render_vscode_vars(program)
        args = [render_vscode_vars(arg) for arg in args]
        cmd = [sys.executable, str(program)] + args

        rich.print(f"[blue bold]Rendered Command for Config `{name}`:[/]", cmd)
        os.execvp(sys.executable, cmd)

    main = run
    default = list_configs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(Config())
    else:
        fire.Fire(Config(), command="default")
