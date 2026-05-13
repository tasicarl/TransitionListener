"""Scan routines for different sampling strategies.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import multiprocessing
from typing import Iterable

import h5py
import numpy as np
from tqdm import tqdm

from transitionlistener.gridplots import plot2dData
from transitionlistener.lineplots import plot1dData
from transitionlistener.helper_functions import load_potential
from transitionlistener import console
from rich.panel import Panel

from .logging_utils import Logger
from .pipeline import run_TL
from .samplers import scanfunc, sample_and_export_to_csv, rand_worker, rand_listener


def _to_scalar_or_nan(value) -> float:
    """Convert value to a scalar float when possible, otherwise NaN."""
    try:
        arr = np.asarray(value)
    except Exception:
        return float("nan")
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        try:
            return float(np.squeeze(arr))
        except Exception:
            return float("nan")
    return float("nan")


def create_grid(xminmax: tuple, yminmax: tuple, N: int,
                xscale: str, yscale: str) -> tuple[np.ndarray]:
    """Create a regular grid of x/y coordinates."""
    if xscale == "lin":
        xrange = np.linspace(*xminmax, num=N)
    elif xscale == "log":
        xrange = np.logspace(*xminmax, num=N)
    else:
        raise ValueError("xscale must be 'lin' or 'log'")

    if yscale == "lin":
        yrange = np.linspace(*yminmax, num=N)
    elif yscale == "log":
        yrange = np.logspace(*yminmax, num=N)
    else:
        raise ValueError("yscale must be 'lin' or 'log'")

    return xrange, yrange


def line_scan(conf, n_jobs: int):
    """1D scan over only one parameter."""
    outpath = conf.output_path

    if conf.scale == "lin":
        xrange = np.linspace(*conf.minmax, num=conf.N)
    elif conf.scale == "log":
        xrange = np.logspace(*conf.minmax, num=conf.N)
    else:
        raise ValueError("xscale must be 'lin' or 'log'")

    input_params_list = []
    for _, x in enumerate(xrange):
        inputparam_dict = {conf.line_param: x}
        for key, val in conf.other_params.items():
            inputparam_dict[key] = val
        input_params_list.append((inputparam_dict, conf.modelfile,
                                  conf.potential_name, outpath, conf.timeout))

    if n_jobs != 1:
        n_jobs = n_jobs - 1
    pool = multiprocessing.Pool(processes=n_jobs)
    results = []

    for result in tqdm(pool.imap(func=scanfunc,
                                 iterable=input_params_list), total=conf.N):
        results.append(result)

    save_results_1d(
        results,
        conf.N,
        xrange,
        outpath,
        conf.derived_params,
        conf.format,
        conf.line_param,
        conf.other_params,
    )

    if conf.plots:
        plot_results_1d(results, conf, xrange, outpath)


def grid_scan(conf, n_jobs: int):
    """2D grid scan."""
    outpath = conf.output_path
    xrange, yrange = create_grid(conf.minmax["x"], conf.minmax["y"],
                                 conf.N, conf.scales["x"], conf.scales["y"])

    input_params_list = []
    for _, x in enumerate(xrange):
        for _, y in enumerate(yrange):
            inputparam_dict = {}
            for key, val in conf.grid_names.items():
                if key == "x":
                    inputparam_dict[val] = x
                elif key == "y":
                    inputparam_dict[val] = y
            for key, val in conf.other_params.items():
                inputparam_dict[key] = val
            input_params_list.append((inputparam_dict, conf.modelfile,
                                      conf.potential_name, outpath, conf.timeout))

    if n_jobs != 1:
        n_jobs = n_jobs - 1
    pool = multiprocessing.Pool(processes=n_jobs)
    results = []

    for result in tqdm(pool.imap(func=scanfunc,
                                 iterable=input_params_list), total=conf.N**2):
        results.append(result)
    msg = ("Phase transition analysis done. Now, save the "
           "results and check if signal is observable.")
    console.print(Panel.fit(f"[bold green]{msg}[/bold green]", border_style="green"))

    save_results_2d(
        results,
        conf.N,
        xrange,
        yrange,
        outpath,
        conf.derived_params.copy(),
        conf.format,
        conf.grid_names,
        conf.other_params,
    )

    if conf.plots:
        plot_results_2d(results, conf, xrange, yrange, outpath)


def table(conf, n_jobs: int):
    """Evaluate points specified in a CSV table."""
    try:
        file = np.genfromtxt(conf.table_file, delimiter=',')
        file_transpose = np.transpose(file)
    except Exception as exc:
        raise Exception(f"Error reading table file {conf.table_file}: {exc}")

    offset = 1 if np.isnan(file_transpose[0][0]) else 0

    input_params_list = []
    for i in range(conf.N):
        inputparam_dict = {}
        for name in conf.scan_params.keys():
            inputparam_dict[name] = float(file_transpose[conf.scan_params[name]["index"]][i + offset])
        for key, val in conf.other_params.items():
            inputparam_dict[key] = val
        input_params_list.append(
            (
                inputparam_dict,
                conf.modelfile,
                conf.potential_name,
                conf.output_path,
                conf.derived_params,
                conf.timeout,
            )
        )

    if n_jobs != 1:
        n_jobs = n_jobs - 1
    pool = multiprocessing.Pool(processes=n_jobs)
    results = []

    for result in tqdm(pool.imap(func=sample_and_export_to_csv,
                                 iterable=input_params_list), total=conf.N):
        results.append(result)


def random_scan(conf, n_jobs: int):
    """Random parameter scan."""
    N = conf.N

    if n_jobs != 1:
        n_jobs = n_jobs - 1

    manager = multiprocessing.Manager()
    q = manager.Queue()
    pool = multiprocessing.Pool(processes=n_jobs)
    watcher = multiprocessing.Process(target=rand_listener,
                                      args=(q, conf.output_path,
                                            conf.derived_params,))
    # Make sure the I/O worker is running
    watcher.start()

    input_params_list = []
    for _ in range(N):
        inputparam_dict = {}
        cube = np.random.rand(len(conf.scan_params))
        input_params_untransformed = _random_prior_transform(cube, conf)
        for i, name in enumerate(conf.scan_params.keys()):
            inputparam_dict[name] = input_params_untransformed[i]
        for key, val in conf.other_params.items():
            inputparam_dict[key] = val
        input_params_list.append(
            (
                inputparam_dict,
                conf.modelfile,
                conf.potential_name,
                conf.output_path,
                conf.derived_params,
                conf.timeout,
            )
        )

    jobs = []
    for i in range(N):
        job = pool.apply_async(rand_worker, (input_params_list[i], q))
        jobs.append(job)

    for job in jobs:
        job.get()

    q.put('kill')
    pool.close()
    pool.join()


def _random_prior_transform(cube: np.ndarray, conf) -> np.ndarray:
    """Transforms the unit cube to the prior space defined in the configuration."""
    params = cube.copy()
    for i, name in enumerate(conf.scan_params.keys()):
        prior = conf.scan_params[name]["range"]
        try:
            low = float(prior[0])
        except ValueError:
            low = eval(prior[0])
        try:
            high = float(prior[1])
        except ValueError:
            high = eval(prior[1])

        if conf.scan_params[name]["scale"] == "log":
            params[i] = 10 ** (np.log10(low) + cube[i] * (np.log10(high) - np.log10(low)))
        else:
            params[i] = low + cube[i] * (high - low)
    return params


def save_results_2d(res_dict: Iterable[dict], N: int, xrange: np.ndarray,
                    yrange: np.ndarray, outpath: str, outnames: dict,
                    format: str, grid_names: dict, other_params: dict):
    """Save the result list to a hdf5 or txt file."""
    outnames.update({"error": "error"})
    h = np.array(res_dict).reshape(N, N)
    res = np.full((N, N, len(outnames)), np.nan)

    for i in range(N):
        for j in range(N):
            for k, name in enumerate(outnames):
                if "strongestTransitionObservables" in h[i, j]:
                    if name in h[i, j]["strongestTransitionObservables"]:
                        val = h[i, j]["strongestTransitionObservables"][name]
                        res[i, j, k] = _to_scalar_or_nan(val)
            if "error" in h[i, j]:
                res[i, j, k] = _to_scalar_or_nan(h[i, j]["error"])

    if format == "hdf5":
        with h5py.File(outpath + "data.hdf5", "w") as f:
            f.attrs["xname"] = grid_names["x"]
            f.attrs["yname"] = grid_names["y"]
            f.attrs["xrange"] = xrange
            f.attrs["yrange"] = yrange
            for name, par in zip(other_params.keys(), other_params.values()):
                f.attrs[name] = par

            d = f.create_group('data')
            for on in outnames:
                d.create_dataset(on, shape=(N, N))

            for i, x in enumerate(xrange):
                for j, y in enumerate(yrange):
                    for k, on in enumerate(outnames):
                        d[on][i, j] = res[i, j, k]
    elif format == "txt":
        for k, on in enumerate(outnames):
            data = np.zeros((N, N))
            for i, x in enumerate(xrange):
                for j, y in enumerate(yrange):
                    data[i, j] = res[i, j, k]
            np.savetxt(outpath + on + ".txt", data)

        xname = grid_names["x"]
        yname = grid_names["y"]
        np.savetxt(outpath + "xrange.txt", xrange,
                   header=f"X input range for parameter: {xname:}")
        np.savetxt(outpath + "yrange.txt", yrange,
                   header=f"Y input range for parameter: {yname:}")

        with open(outpath + "other_params.txt", "w") as fh:
            fh.write("# " + "\t".join(other_params.keys()) + "\n")
            fh.write("\t".join(str(v) for v in other_params.values()) + "\n")


def plot_results_2d(results_dict: Iterable[dict], conf, xrange: np.ndarray, yrange: np.ndarray, outpath: str):
    """Make the grid plots."""
    derived_param_names = [name for name in conf.derived_params.keys()]
    derived_param_plot_names = [name for name in conf.derived_params.values()]
    grid_params_names = [conf.grid_names["x"], conf.grid_names["y"]]
    grid_params_plot_names = [conf.model_params[name]["plotname"] for name in grid_params_names]
    overview_param_plot_names = [
        plot_name for name, plot_name in zip(derived_param_names, derived_param_plot_names)
        if name in conf.overview_param_names
    ]

    scale = conf.scales["x"] + conf.scales["y"]
    all_params = np.array(results_dict).reshape(conf.N, conf.N)

    grid = plot2dData(
        all_params,
        x=xrange,
        y=yrange,
        scale=scale,
        xy_plot_names=grid_params_plot_names,
        foldername=outpath,
        derived_params_names=derived_param_names,
        derived_params_plot_names=derived_param_plot_names,
        overview_title=conf.plot_description,
        show_scan_points=conf.show_scan_points,
    )
    def _run_plot_step(step_name: str, fn):
        try:
            fn()
        except Exception as exc:
            msg = f"[yellow]Skipping plot step '{step_name}': {exc}[/yellow]"
            console.print(Panel.fit(msg, border_style="yellow"))

    # Produce the paper-critical four-panel overview first.
    _run_plot_step(
        "overview_plot",
        lambda: grid.plot_overview(
            overview_detector_name=conf.overview_detector_name,
            overview_param_names=conf.overview_param_names,
            overview_param_plot_names=overview_param_plot_names,
        ),
    )
    _run_plot_step("log_params", grid.plot_log_params)
    _run_plot_step("lin_params", grid.plot_lin_params)
    _run_plot_step("save_SNRs", grid.save_SNRs)
    _run_plot_step("plot_SNRs", grid.plot_SNRs)
    _run_plot_step("save_logLs", grid.save_logLs)
    _run_plot_step("plot_logLs", grid.plot_logLs)
    _run_plot_step("save_add_infos", grid.save_add_infos)
    _run_plot_step("plot_add_infos", grid.plot_add_infos)
    _run_plot_step("plot_mass_spectrum", grid.plot_mass_spectrum)
    _run_plot_step("plot_errors", grid.plot_errors)


def save_results_1d(res_dict: Iterable[dict], N: int, xrange: np.ndarray, outpath: str,
                    outnames: dict, format: str, line_param: str, other_params: dict):
    """Save the result list to a hdf5 or txt file."""
    outnames.update({"error": "error"})
    h = np.array(res_dict).reshape(N)
    res = np.full((N, len(outnames)), np.nan)

    for i in range(N):
        for k, name in enumerate(outnames):
            if "strongestTransitionObservables" in h[i]:
                if name in h[i]["strongestTransitionObservables"]:
                    val = h[i]["strongestTransitionObservables"][name]
                    res[i, k] = _to_scalar_or_nan(val)
        if "error" in h[i]:
            res[i, k] = _to_scalar_or_nan(h[i]["error"])

    if format == "hdf5":
        with h5py.File(outpath + "data.hdf5", "w") as f:
            f.attrs["xname"] = line_param
            f.attrs["xrange"] = xrange
            for name, par in zip(other_params.keys(), other_params.values()):
                f.attrs[name] = par

            d = f.create_group('data')
            for on in outnames:
                d.create_dataset(on, shape=(N,))

            for i, x in enumerate(xrange):
                for k, on in enumerate(outnames):
                    d[on][i] = res[i, k]
    elif format == "txt":
        for k, on in enumerate(outnames):
            data = np.zeros((N,))
            for i, x in enumerate(xrange):
                data[i] = res[i, k]
            np.savetxt(outpath + on + ".txt", data)

        np.savetxt(outpath + "xrange.txt", xrange,
                   header=f"X input range for parameter: {line_param:}")
        with open(outpath + "other_params.txt", "w") as fh:
            fh.write("# " + "\t".join(other_params.keys()) + "\n")
            fh.write("\t".join(str(v) for v in other_params.values()) + "\n")


def plot_results_1d(results_dict: Iterable[dict], conf, xrange: np.ndarray, outpath: str):
    """Generate line scan plots."""
    derived_param_names = [name for name in conf.derived_params.keys()]
    derived_param_plot_names = [name for name in conf.derived_params.values()]
    x_plot_name = conf.model_params[conf.line_param]["plotname"]
    overview_param_plot_names = [
        plot_name for name, plot_name in zip(derived_param_names, derived_param_plot_names)
        if name in conf.overview_param_names
    ]

    scale = conf.scale
    all_params = np.array(results_dict).reshape(conf.N)

    grid = plot1dData(
        all_params,
        x=xrange,
        scale=scale,
        x_plot_name=x_plot_name,
        foldername=outpath,
        derived_params_names=derived_param_names,
        derived_params_plot_names=derived_param_plot_names,
        overview_title=conf.plot_description,
        show_scan_points=conf.show_scan_points,
    )
    def _run_plot_step(step_name: str, fn):
        try:
            fn()
        except Exception as exc:
            msg = f"[yellow]Skipping plot step '{step_name}': {exc}[/yellow]"
            console.print(Panel.fit(msg, border_style="yellow"))

    # Produce the paper-critical four-panel overview first.
    _run_plot_step(
        "overview_plot",
        lambda: grid.plot_overview(
            overview_detector_name=conf.overview_detector_name,
            overview_param_names=conf.overview_param_names,
            overview_param_plot_names=overview_param_plot_names,
        ),
    )
    _run_plot_step("plot_log_params", grid.plot_log_params)
    _run_plot_step("plot_lin_params", grid.plot_lin_params)
    _run_plot_step("save_SNRs", grid.save_SNRs)
    _run_plot_step("plot_SNRs", grid.plot_SNRs)
    _run_plot_step("save_logLs", grid.save_logLs)
    _run_plot_step("plot_logLs", grid.plot_logLs)
    _run_plot_step("save_add_infos", grid.save_add_infos)
    _run_plot_step("plot_add_infos", grid.plot_add_infos)
    _run_plot_step("plot_mass_spectrum", grid.plot_mass_spectrum)
    _run_plot_step("plot_errors", grid.plot_errors)
