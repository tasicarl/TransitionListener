"""Command-line interface for TransitionListener.

This module exposes the ``tl`` entry point which is used to launch scans and
other analyses from the terminal.  The command line interface is intentionally
minimal – most of the configuration is supplied through YAML files – but the
help text should nonetheless offer a concise overview of what the tool does and
how it can be used.  The updates in this module focus on surfacing that
information so that ``tl -h`` is informative for new users.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import argparse
import os
import textwrap
import time

from transitionlistener import console
from rich.panel import Panel

from . import state
from .config_loader import ScanConfig
from .scans import grid_scan, line_scan, random_scan, table
from .single_point import single
from .samplers import ultranest


def main():
    """Console entry point."""
    t0 = time.time()

    parser = argparse.ArgumentParser(
        prog="tl",
        description=textwrap.dedent(
            """
            TransitionListener performs parameter scans and single point
            analyses for cosmological phase transition and gravitational
            wave calculations.  Provide a YAML configuration file describing
            the scan type, model inputs, and output location to launch a run.
            """
        ).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Common scan types include GridScan, LineScan, RandomScan, SinglePoint,
            UltranestScan, and TableScan. The scan type is selected in the YAML
            configuration file and determines which numerical routine is
            executed.

            Examples:
              tl -c examples/example_point.yaml -v True
              tl -c examples/example_grid.yaml -j 10
              mpiexec -n 10 tl -c examples/example_ultranest.yaml

            Documentation: https://www.tasillo.de/transitionlistener_development/
            """
        ).strip(),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes to use for supported scan "
            "types (defaults to 1)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE.yaml",
        help=(
            "Path to the YAML configuration file that defines the scan. The "
            "file specifies the scan type, physics inputs, and output path."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed logging to the console.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode (do not suppress runtime warnings).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Show the TransitionListener version and exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a quick installation self-test and exit.",
    )

    args = parser.parse_args()

    if args.check:
        from .check import run_checks
        raise SystemExit(run_checks())

    if args.version:
        from transitionlistener import __version__
        console.print(Panel.fit(f"[bold green]TransitionListener version {__version__}[/bold green]", border_style="green"))
        raise SystemExit(0)

    if not args.config:
        parser.error("a configuration file must be provided with -c/--config")

    state.set_debugmode(args.debug)

    config_path = os.path.expanduser(args.config)
    if not os.path.exists(config_path):
        parser.error(f"configuration file not found: {config_path}")

    state.config = ScanConfig(config_path)
    os.makedirs(state.config.output_path, exist_ok=True)

    mpi_rank_env_vars = [
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "SLURM_PROCID",
        "RANK",
        "I_MPI_RANK",
        "MPI_LOCALRANKID",
    ]
    mpi_size_env_vars = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "SLURM_NPROCS",
        "SLURM_NTASKS",
        "WORLD_SIZE",
        "MPIEXEC_TIMEOUT",
    ]

    rank_value = None
    for env_var in mpi_rank_env_vars:
        value = os.environ.get(env_var)
        if value is not None:
            rank_value = value
            break

    rank_from_env = rank_value is not None
    if rank_value is None:
        rank_value = "0"

    try:
        rank = int(rank_value)
    except Exception:
        rank = 0

    is_mpi = rank_from_env or any(os.environ.get(env_var) is not None for env_var in mpi_size_env_vars)
    if (not is_mpi) or (rank == 0):
        message = "[green bold]Welcome to TransitionListener v2.0![/]"
        console.print(Panel.fit(message, border_style="green"))
        console.print()
        state.config.display()
        message = f"[green]Starting analysis of {state.config.description}[/]"
        console.print(Panel.fit(message, border_style="green"))
        console.print()

    try:
        if state.config.type == "GridScan":
            grid_scan(state.config, n_jobs=args.jobs)
        elif state.config.type == "LineScan":
            line_scan(state.config, n_jobs=args.jobs)
        elif state.config.type == "RandomScan":
            random_scan(state.config, n_jobs=args.jobs)
        elif state.config.type == "SinglePoint":
            single(state.config, verbose=args.verbose)
        elif state.config.type == "UltranestScan":
            if args.jobs != 1:
                msg = ("Error: Ultranest does not support parallelization through "
                       "the -j option. We recommend parallelizing it using MPI. "
                       "For this you need to execute: \n\n"
                       "mpiexec -n <num_procs> tl -c <config.yaml>\n"
                       "or, if `tl` is not installed,\n"
                       "mpiexec -n <num_procs> env PYTHONPATH=src "
                       "python -m transitionlistener.interface.cli -c <config.yaml>")
                console.print(Panel.fit(f"[bold red]{msg}[/bold red]", border_style="red"))
                raise SystemExit(1)
            ultranest(state.config)
        elif state.config.type == "TableScan":
            table(state.config, n_jobs=args.jobs)
        else:
            raise ValueError("Unknown scan type.")
    except ValueError as err:
        console.print(Panel.fit(str(err), border_style="red"))
        raise SystemExit(1) from err

    tend = time.time()
    if (not is_mpi) or (rank == 0):
        msg = (f"Finished in {time.strftime('%H:%M:%S', time.gmtime(tend-t0))}")
        console.print(Panel.fit(f"[bold green]{msg}[/bold green]", border_style="green"))
