# /// script
# dependencies = [
#   "typer >=0.21, <1",
#   "jinja2 >=3.1, <4",
#   "structlog >=25, <26",
# ]
# ///
"""
Terragrunt stack utilities with a compact CLI.

Provides plan/apply/report commands, supports explicit stack root path,
streams output, and summarizes per-unit changes from plan logs.
"""

from __future__ import annotations

import re
import subprocess as sp
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import IO, Annotated, TextIO

import jinja2
import structlog
import typer

# App setup
app = typer.Typer(
    name="tg",
    help="Terragrunt stack ops: plan, apply, report",
    add_completion=False,
    no_args_is_help=True,
)


# Types and constants
class ChangeState(str, Enum):
    CLEAN = "NO_CHANGES"
    DIRTY = "HAS_CHANGES"
    ERROR = "HAS_ERROR"


@dataclass
class StackDiffs:
    """Per-unit diff summary extracted from plan logs."""

    stable: list[str]
    diffs: dict[str, list[str]]


@dataclass
class CommandResult:
    code: int
    stdout: str
    stderr: str


TOFU_RE = re.compile(r".*\[(?P<module>[^\]]+)\]\s+tofu:\s(?P<message>.*)")

CLR = {
    "GREEN": "\033[0;32m",
    "YELLOW": "\033[0;33m",
    "RED": "\033[0;31m",
    "RESET": "\033[0m",
}

GITHUB_PR_COMMENT_TMPL = """
{% if state == 'NO_CHANGES' -%}
No changes detected.
{% elif state == 'HAS_ERROR' -%}
Errors found. See logs.
{% else -%}

Unchanged units:

{% if diffs and diffs.stable -%}
{% for unit in diffs.stable -%}
- `{{ unit }}`
{% endfor -%}
{% else -%}
None
{% endif %}


Changed units:

{% if diffs and diffs.diffs -%}
{% for unit, lines in diffs.diffs.items() -%}

- `{{ unit }}`

<details>
<summary>Changes to {{ unit }}</summary>

```diff
{% for ln in lines -%}
{{ ln }}
{% endfor -%}
```

</details>

{% endfor -%}
{% else -%}
None
{% endif %}

{% endif %}
"""

LOGGER = structlog.get_logger(__name__)


# Helpers

def _normalize_diff_line(line: str) -> str:
    """Adjust diff markers to suit fenced diff blocks."""
    stripped = line.lstrip()
    replace = {
        "~": "!",
        "+": "+",
        "-": "-",
    }
    for k, v in replace.items():
        if stripped.startswith(k):
            return v + line.replace(stripped, stripped[1:], 1)
    return line


def _collect_unit_entries(log_file: Path) -> dict[str, list[str]]:
    """Group tofu messages by unit from a plan log file."""
    entries: dict[str, list[str]] = defaultdict(list)
    with log_file.open("r", encoding="utf-8") as fh:
        for ln in fh:
            m = TOFU_RE.match(ln)
            if not m:
                continue
            unit = m.group("module").replace(".terragrunt-stack/", "")
            entries[unit].append(m.group("message"))
    return entries


def _summarize_unit_logs(log_file: Path) -> StackDiffs | None:
    """Produce a unit-by-unit summary from a plan log file."""
    if not log_file.exists():
        LOGGER.warning("log file missing", path=str(log_file))
        return None

    grouped = _collect_unit_entries(log_file)
    stable: list[str] = []
    diffs: dict[str, list[str]] = {}

    for unit, msgs in grouped.items():
        if any(msg.strip().lower().startswith("no changes") for msg in msgs):
            stable.append(unit)
        else:
            diffs[unit] = [_normalize_diff_line(m) for m in msgs]

    return StackDiffs(stable=stable, diffs=diffs)


def _write_exit_marker(path: Path, code: int) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"terragrunt-exit-code={code}\n")


def run_live(
    cmd: list[str], *, log_file: Path | None = None, cwd: Path | None = None, quiet: bool = False
) -> CommandResult:
    """Execute a command, stream output, optionally tee to a log file."""
    proc = sp.Popen(  # noqa: S603
        cmd,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        bufsize=1,
        cwd=str(cwd) if cwd else None,
    )

    out_lines: list[str] = []
    err_lines: list[str] = []

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()
        log_fh = log_file.open("a", encoding="utf-8")
    else:
        log_fh = None

    def _drain(pipe: IO[str], collector: list[str], target: TextIO) -> None:
        try:
            for ln in iter(pipe.readline, ""):
                collector.append(ln)
                if not quiet:
                    target.write(ln)
                    target.flush()
                if log_fh is not None:
                    log_fh.write(ln)
                    log_fh.flush()
        finally:
            pipe.close()

    t_out = Thread(target=_drain, args=(proc.stdout, out_lines, sys.stdout), daemon=True)
    t_err = Thread(target=_drain, args=(proc.stderr, err_lines, sys.stderr), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    if log_fh is not None:
        log_fh.flush()
        log_fh.close()

    return CommandResult(code=proc.returncode, stdout="".join(out_lines), stderr="".join(err_lines))


# Commands
@app.command("plan-github-pr-comment")
def plan_github_pr_comment(
    log_file: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to the generated plan log file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            help="Where to write the generated summary.",
        ),
    ],
) -> None:
    """Parse a plan log and render a summary to post to a GitHub PR comment."""
    if not log_file.exists():
        raise typer.BadParameter(f"Log file not found: {log_file}")

    content = log_file.read_text()
    if "terragrunt-exit-code=" not in content:
        raise typer.BadParameter("Log file missing terragrunt-exit-code marker")

    if "terragrunt-exit-code=0" in content:
        state = "NO_CHANGES"
    elif "terragrunt-exit-code=1" in content:
        state = "HAS_ERROR"
    elif "terragrunt-exit-code=2" in content:
        state = "HAS_CHANGES"
    else:
        raise typer.BadParameter("Unexpected exit code marker in log file")

    diffs = _summarize_unit_logs(log_file)
    template = jinja2.Template(GITHUB_PR_COMMENT_TMPL)
    rendered = template.render(
        state=state,
        diffs=diffs,
    )
    output.write_text(rendered)
    typer.echo(f"Wrote GitHub PR comment format to {output}")


class Runner:
    def __init__(self, stack_root: Path) -> None:
        self.stack_root = stack_root

    def plan(self, *, log_file: Path | None = None) -> ChangeState:
        if log_file and log_file.exists():
            log_file.unlink()

        plan_cmd = [
            "terragrunt",
            f"--working-dir={self.stack_root!s}",
            "stack",
            "run",
            "--",
            "plan",
            "-detailed-exitcode",
            "-out=tofu.plan",
        ]

        result = run_live(plan_cmd, log_file=log_file)

        if log_file and result.code != 1:
            show_cmd = [
                "terragrunt",
                "--no-color",
                f"--working-dir={self.stack_root!s}",
                "stack",
                "run",
                "--",
                "show",
                "tofu.plan",
            ]
            # Avoid echoing show to console to reduce duplicate lines
            run_live(show_cmd, log_file=log_file, quiet=True)
            _write_exit_marker(log_file, result.code)

        if result.code == 0:
            typer.echo(f"✅ {CLR['GREEN']}Plan is clean: no changes.{CLR['RESET']}")
            return ChangeState.CLEAN
        if result.code == 2:
            typer.echo(f"⚠️ {CLR['YELLOW']}Plan indicates pending changes. See logs if enabled.{CLR['RESET']}")
            return ChangeState.DIRTY

        if not log_file and result.stderr:
            _ = sys.stderr.write(result.stderr)
        typer.echo(f"❌ {CLR['RED']}Terragrunt plan failed. Check the output above or logs.{CLR['RESET']}")
        raise typer.Exit(code=1)

    def apply(self, *, non_interactive: bool = False) -> None:
        tg_args = [f"--working-dir={self.stack_root!s}"]
        if non_interactive:
            tg_args.extend(["--non-interactive"])
        cmd = [
            "terragrunt",
            *tg_args,
            "stack",
            "run",
            "--",
            "apply",
        ]
        result = run_live(cmd)
        if result.code != 0:
            raise typer.Exit(code=result.code)


@app.command("plan")
def plan(
    stack_root: Path = typer.Option(
        ...,
        envvar="TGOPS_STACK_ROOT", 
        help="Path to the terragrunt stack main directory",
    ),
    log_file: Path = typer.Option(
        ...,
        help="Optional path to output logs to.",
        envvar="TGOPS_LOG_FILE",
    ),
) -> None:
    """Run a detailed-exitcode plan for the provided stack."""
    Runner(stack_root=stack_root).plan(log_file=log_file)


@app.command("apply")
def apply(
    stack_root: Path = typer.Option(
        ...,
        envvar="TGOPS_STACK_ROOT",
        help="Path to the terragrunt stack main directory",
    ),
    *,
    non_interactive: bool = typer.Option(
        False,
        help="If set, write plan + show output and exit marker to logs/<env>.plan.log",
        envvar="TGOPS_NON_INTERACTIVE",
    ),
) -> None:
    """Apply the terragrunt stack at the provided path."""
    Runner(stack_root=stack_root).apply(non_interactive=non_interactive)


if __name__ == "__main__":
    app()
