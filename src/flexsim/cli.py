"""Shared plumbing for the ``run/*.py`` entry points.

Each entry point is a table of subcommands. A subcommand names the module whose
``main()`` does the work; everything after the subcommand is passed through to
that module's own argument parser, so ``run/plots.py susc --help`` shows the
real options.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field

from flexsim.paths import OUTPUT_ROOT, enter_output_root


@dataclass(frozen=True)
class Command:
    module: str                       # e.g. "flexsim.susc.plot"
    help: str
    prefix: list[str] = field(default_factory=list)  # args inserted before the user's


def _usage(prog: str, description: str | None, commands: dict[str, Command]) -> str:
    width = max(len(name) for name in commands)
    lines = [description or "", "", f"usage: python run/{prog} <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {cmd.help}" for name, cmd in commands.items()]
    lines += ["", f"Run 'python run/{prog} <command> --help' for a command's options.",
              f"Outputs go under {OUTPUT_ROOT} (override with FLEX_OUTPUT_ROOT)."]
    return "\n".join(lines)


def dispatch(prog: str, description: str | None, commands: dict[str, Command],
             argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(prog, description, commands))
        sys.exit(0 if argv else 2)

    name, rest = argv[0], argv[1:]
    if name not in commands:
        print(f"unknown command '{name}'\n\n" + _usage(prog, description, commands), file=sys.stderr)
        sys.exit(2)

    cmd = commands[name]
    module = importlib.import_module(cmd.module)
    enter_output_root()
    # With a prefix the module's own subparser adds the subcommand name to usage lines.
    shown = f"run/{prog}" if cmd.prefix else f"run/{prog} {name}"
    sys.argv = [shown, *cmd.prefix, *rest]
    result = module.main()
    if isinstance(result, int):
        sys.exit(result)
