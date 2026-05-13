"""Sampler integrations and helpers for TransitionListener.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Dict
import numpy as np
import pandas as pd

from ultranest import ReactiveNestedSampler

from transitionlistener.observability import PTA_LIKELIHOOD_SETTINGS
from transitionlistener.helper_functions import import_file, load_potential

from . import state
from .logging_utils import Logger
from .output_schema import OutputSchema, OutputRandomscan
from .pipeline import run_TL


PTA_LABELS = tuple(PTA_LIKELIHOOD_SETTINGS.keys())
MPI_RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
    "RANK",
    "I_MPI_RANK",
    "MPI_LOCALRANKID",
)


def scanfunc(input, verbose: bool = False) -> dict:
    """Wrapper used by worker processes to execute a single scan point."""
    inputparams_dict, modelfile, potential_name, outpath, timeout = input
    potential = load_potential(modelfile, potential_name)
    errorlogger = Logger(outpath + "error.log")
    resultlogger = Logger(outpath + "result.log")
    return run_TL(
        inputparams_dict,
        potential,
        errorlogger,
        resultlogger,
        timeout,
        verbose=verbose,
        call_from_sampler=False,
        include_smbhb=False,
    )


def sample_and_export_to_csv(args):
    """Evaluate a single parameter point and append the
    outcome to ``output_table.csv``."""
    inputparams_dict, modelfile, potential_name, output_path, derived_params, timeout = args

    potential = load_potential(modelfile, potential_name)
    errorlogger = Logger(output_path + "error.log")
    resultlogger = Logger(output_path + "result.log")

    res = call_by_sampler(potential, inputparams_dict, errorlogger,
                          resultlogger, timeout, verbose=False, include_smbhb=False)
    res_params = extract_parameters(res, derived_params)

    schema = OutputSchema(potential, derived_params.keys())
    output_file = _ranked_output_file(output_path + "output_table.csv")
    schema.append_row(output_file, inputparams_dict, res_params)
    
    # Return only the PTArcade log-likelihood value for sampler use
    lnL_value = res_params.get("PTArcade_lnL", np.nan)
    return pd.Series([lnL_value], name="lnL")


def rand_listener(q, output_path, derived_params) -> None:
    """Listener which appends the output to a file.
    

    Parameters
    ----------
    q : multiprocessing.Manager.Queue
    output_path : str
        Path to the output. The output file is ouput_path + output_table.csv
    derived_params : dict

    Returns
    -------
    None."""
    output_file = output_path + "output_table.csv"
    outputter = OutputRandomscan(output_file)
    while True:
        message = q.get()
        if message == 'kill':
            outputter.stop()
            break
        try:
            inputparams_dict, res_params = message
            outputter.write_row(inputparams_dict, res_params)
        except Exception as e:
            print(e)


def rand_worker(inp_params, q):
    """Worker for the random scan.

    Puts the results in the queue.

    Parameters
    ----------
    inp_params : tuple
        Tuple with all the input parameters
    q : multiprocessing.Manager.Queue

    Returns
    -------
    None."""
    inputparams_dict, modelfile, potential_name, output_path, derived_params, timeout = inp_params

    potential = load_potential(modelfile, potential_name)
    errorlogger = Logger(output_path + "error.log")
    resultlogger = Logger(output_path + "result.log")

    res = call_by_sampler(potential, inputparams_dict, errorlogger,
                          resultlogger, timeout, verbose=False, include_smbhb=False)

    # this is a bad temporary fix.
    # However I think generalising this would require a lot of refactoring
    # Dont return the masses, that makes everything complicated. They can
    # be computed easily after the scans.
    if res["error"] != 0:
        errcode = res["error"]
        errms = res["errormsg"]
        res_params = get_empty_result()
        res_params["error"] = errcode
        res_params["errormsg"] = ""
    else:
        res_params = extract_parameters(res, derived_params)

    res = (inputparams_dict, res_params)
    q.put(res)
    return res


def call_by_sampler(potential: object, inputparams_dict: dict,
                    errorlogger: Logger, resultlogger: Logger,
                    timeout: float, verbose: bool = False, include_smbhb: bool = False):
    """Evaluate TransitionListener for external samplers and return the result dictionary."""
    res = {'error': 0}
    res['errormsg'] = ""
    state.TIMEOUT = int(timeout)

    if state.DEBUGMODE:
        run_TL(
            inputparams_dict,
            potential,
            errorlogger,
            resultlogger,
            timeout,
            verbose=verbose,
            call_from_sampler=True,
            include_smbhb=include_smbhb,
        )

    try:
        all_params_dict = run_TL(
            inputparams_dict,
            potential,
            errorlogger,
            resultlogger,
            timeout,
            verbose=verbose,
            call_from_sampler=True,
            include_smbhb=include_smbhb,
        )
        res["all_params_dict"] = all_params_dict

    except Exception as err:
        if hasattr(err, 'errorcode') and err.errorcode is not None:
            line = (
                "ERROR in TransitionListener of type "
                + str(err.errorcode)
                + " for input parameters "
                + str(inputparams_dict)
                + ": "
                + getattr(err, 'message', str(err))
            )
            res['errormsg'] = line
            res['error'] = err.errorcode
        else:
            line = (
                "ERROR in TransitionListener of type -1 for input parameters "
                + str(inputparams_dict)
                + ": "
                + getattr(err, 'message', str(err))
            )
            res['errormsg'] = line
            res['error'] = -1
    return res


def extract_parameters(res: dict, derived_params: dict, mainPTA: str = "NG15_14bins") -> Dict[str, float]:
    """Extract the parameters from the result dictionary."""

    def _to_primitive(value):
        """Convert numpy scalars and arrays to Python primitives for CSV output."""
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (float, int, bool, str)):
            return value
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return _to_primitive(value.flatten()[0])
            return np.nan
        if value is None:
            return np.nan
        return value

    params: Dict[str, float] = {}

    error_value = res.get("error", np.nan)
    if isinstance(error_value, np.ndarray):
        error_value = error_value.flatten()[0] if error_value.size else np.nan
    params["error"] = _to_primitive(error_value)
    params["errormsg"] = str(res.get("errormsg", ""))
    params["PTArcade_lnL"] = -np.inf
    params["smoothened_lnL"] = -np.inf
    params["mock_lnL"] = -np.inf
    params["lnL"] = -np.inf

    if params["error"] == 0:
        all_params_dict = res.get("all_params_dict", {})
        strongest = all_params_dict.get("strongestTransitionObservables", {})
        observability = all_params_dict.get("observability", {})

        for name in derived_params.keys():
            if name in strongest:
                params[name] = _to_primitive(strongest[name])
            elif name in observability:
                params[name] = _to_primitive(observability[name])
            else:
                params[name] = np.nan

        main_pta_key = f"lnL_PTArcade_{mainPTA}"
        params["PTArcade_lnL"] = _to_primitive(
            observability.get(
                main_pta_key,
                observability.get(f"lnL_{mainPTA}", -np.inf),
            )
        )
        smooth_key = f"lnL_smooth_{mainPTA}"
        params["smoothened_lnL"] = _to_primitive(
            observability.get(smooth_key, -np.inf)
        )
        mock_key = f"lnL_mock_{mainPTA}"
        params["mock_lnL"] = _to_primitive(
            observability.get(mock_key, -np.inf)
        )
        params["lnL"] = params["PTArcade_lnL"]

        for label in PTA_LABELS:
            smooth_src = f"lnL_smooth_{label}"
            if smooth_src in observability:
                params[f"lnL_smoothened_{label}"] = _to_primitive(observability[smooth_src])

            ptarcade_src = f"lnL_PTArcade_{label}"
            if ptarcade_src in observability:
                params[f"lnL_PTArcade_{label}"] = _to_primitive(observability[ptarcade_src])

            mock_src = f"lnL_mock_{label}"
            if mock_src in observability:
                params[f"lnL_mock_{label}"] = _to_primitive(observability[mock_src])

        lin_renames = {
            "f_peak_Hz_lin": "f_peak_Hz",
            "h2OmegaGW_peak_lin": "h2OmegaGW_peak",
            "f_pivot_Hz_lin": "f_pivot_Hz",
            "h2OmegaGW_at_pivot_lin": "h2OmegaGW_at_pivot",
        }
        skip_exact = {
            "DNeff_GW_lin",
            "DNeff_GW_log10",
            "DNeff_GW_latex",
        }

        for key, value in observability.items():
            if key in params:
                continue

            if key in lin_renames:
                params[lin_renames[key]] = _to_primitive(value)
                continue

            if key in skip_exact:
                continue
            if key.endswith("_detectable"):
                continue
            if key.endswith("_latex") or key.endswith("_log10"):
                continue
            if key.startswith("lnL_") and not key.startswith("lnL_PTArcade_"):
                continue
            if key.startswith("delta_lnL_") or key.startswith("sigma_") or key.startswith("within_3sigma_"):
                continue

            params[key] = _to_primitive(value)
    return params


def log_likelihood(args):
    """Evaluate the nested-sampling log-likelihood for a given parameter vector."""
    inputparams_dict = {name: val for name, val in zip(state.config.scan_params, args)}

    transformation_file = getattr(state.config, "transformation_file", None)
    if transformation_file is not None:
        transform_module = import_file(transformation_file)
        transform_scan_params = transform_module.transform_scan_params
        inputparams_dict = transform_scan_params(inputparams_dict)

    if state.config.other_params is not None:
        inputparams_dict.update(state.config.other_params)

    res = call_by_sampler(
        state.potential,
        inputparams_dict,
        state.errorlogger,
        state.resultlogger,
        state.TIMEOUT,
        verbose=False,
        include_smbhb=False,
    )
    params = extract_parameters(res, state.config.derived_params)

    schema = OutputSchema(state.model, state.config.derived_params.keys())
    output_file = _ranked_output_file(state.config.output_path + "output_table.csv")
    schema.append_row(output_file, inputparams_dict, params)

    if params["error"] != 0:
        return -1e100
    smooth_value = params.get("smoothened_lnL", -np.inf)
    if not np.isfinite(smooth_value):
        return -1e100
    return smooth_value


def prior_transform(cube):
    """Transforms the unit cube to the prior space."""
    params = cube.copy()
    for i, name in enumerate(state.config.scan_params.keys()):
        prior = state.config.scan_params[name]["range"]

        try:
            low = float(prior[0])
        except ValueError:
            low = eval(prior[0])

        try:
            high = float(prior[1])
        except ValueError:
            high = eval(prior[1])

        if state.config.scan_params[name]["scale"] == "log":
            params[i] = 10 ** (np.log10(low) + cube[i] * (np.log10(high) - np.log10(low)))
        else:
            params[i] = low + cube[i] * (high - low)
    return params


def ultranest(conf):
    """Run the Ultranest sampler."""
    _prepare_ultranest_output_dir(conf)
    state.errorlogger = Logger(conf.output_path + "error.log")
    state.resultlogger = Logger(conf.output_path + "result.log")
    state.TIMEOUT = conf.timeout
    state.config = conf
    points_file = os.path.join(conf.output_path, "results", "points.hdf5")
    try:
        sampler = ReactiveNestedSampler(
            list(conf.scan_params.keys()),
            log_likelihood,
            prior_transform,
            log_dir=conf.output_path,
            resume=conf.resume_mode,
            vectorized=False,
            draw_multiple=False,
            ndraw_min=1,
            ndraw_max=1,
        )
    except OSError as exc:
        if (
            conf.resume_mode in (True, "resume", "resume-similar")
            and os.path.exists(points_file)
            and "truncated file" in str(exc)
        ):
            raise ValueError(
                "UltraNest could not resume because the stored point file is corrupted: "
                f"{points_file}. Delete that file or set `resume_mode: overwrite` in the "
                "YAML to start a fresh run."
            ) from exc
        raise

    sampler.run(
        min_num_live_points=200,
        dKL=25.0,
        max_ncalls=300_000,
        min_ess=2000,
        frac_remain=0.01,
        dlogz=5,
        log_interval=1,
        show_status=True,
    )


def _prepare_ultranest_output_dir(conf) -> None:
    """Clean UltraNest-generated output files when starting a fresh run."""
    if conf.resume_mode != "overwrite":
        return

    rank = _current_mpi_rank()
    if rank == 0:
        files_to_remove = (
            "debug.log",
            "error.log",
            "result.log",
            "output_table.csv",
        )
        for filename in files_to_remove:
            path = os.path.join(conf.output_path, filename)
            if os.path.exists(path):
                os.remove(path)

        for path in glob.glob(os.path.join(conf.output_path, "output_table_rank*.csv")):
            if os.path.isfile(path):
                os.remove(path)

        for dirname in ("results", "chains", "plots"):
            path = os.path.join(conf.output_path, dirname)
            if os.path.isdir(path):
                shutil.rmtree(path)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            comm.Barrier()
    except Exception:
        pass


def _current_mpi_rank() -> int:
    """Best-effort detection of the current MPI rank from environment variables."""
    for env_var in MPI_RANK_ENV_VARS:
        value = os.environ.get(env_var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def _ranked_output_file(base_file: str) -> str:
    """Return a rank-specific filename to avoid concurrent writes."""
    root, ext = os.path.splitext(base_file)
    rank = _current_mpi_rank()
    return f"{root}_rank{rank:03d}{ext}"


def get_empty_result() -> dict:
    """Return an empty result for the random scan.
    This is neccessary when the potential initialisation
    fails

    Parameters
    ----------

    Returns
    -------
    dict :
        The result with np.nans """
    res = {'error': 0,
           'errormsg': '',
           'PTArcade_lnL': np.nan,
           'smoothened_lnL': np.nan,
           'mock_lnL': np.nan,
           'lnL': np.nan,
           'alpha': np.nan,
           'alpha_theta': np.nan,
           'alpha_thetabar': np.nan,
           'alpha_inf': np.nan,
           'alpha_eq': np.nan,
           'betaH_S3': np.nan,
           'betaH_RH': np.nan,
           'RH': np.nan,
           'Treh_SM_GeV': np.nan,
           'Tperc_SM_GeV': np.nan,
           'g_eff_tot_reh': np.nan,
           'h_eff_tot_reh': np.nan,
           'kappa_phi': np.nan,
           'kappa_sw': np.nan,
           'kappa_turb': np.nan,
           'g0': np.nan,
           'h0': np.nan,
           'v_wall': np.nan,
           'D': np.nan,
           'c_s': np.nan,
           'c_s_sym': np.nan,
           'c_s_bro': np.nan,
           'step': np.nan,
           'total_steps': np.nan,
           'Tnuc_SM_GeV': np.nan,
           'Tcrit_SM_GeV': np.nan,
           'Tf_SM_GeV': np.nan,
           'xi_crit': np.nan,
           'WARNING:too_weak_to_compute_perc': False,
           'WARNING:no_perc_splines': False,
           'WARNING:betaH_small': False,
           'WARNING:betaH_very_small': False,
           'WARNING:betaH_mismatch': False,
           'WARNING:betaH_nonfinite': False,
           'WARNING:nucleationRate_nonexponential': False,
           'WARNING:not_T0_global_min': False,
           'lnL_smoothened_NG15_14bins': np.nan,
           'lnL_PTArcade_NG15_14bins': np.nan,
           'lnL_mock_NG15_14bins': np.nan,
           'lnL_smoothened_NG15_5bins': np.nan,
           'lnL_PTArcade_NG15_5bins': np.nan,
           'lnL_mock_NG15_5bins': np.nan,
           'lnL_smoothened_NG12_5bins': np.nan,
           'lnL_PTArcade_NG12_5bins': np.nan,
           'lnL_mock_NG12_5bins': np.nan,
           'lnL_smoothened_IPTA2_13bins': np.nan,
           'lnL_PTArcade_IPTA2_13bins': np.nan,
           'lnL_mock_IPTA2_13bins': np.nan,
           'SKA_5_yrs_SNR': np.nan,
           'SKA_10_yrs_SNR': np.nan,
           'SKA_20_yrs_SNR': np.nan,
           'EPTA_18_yrs_SNR': np.nan,
           'NANOGrav_11_yrs_SNR': np.nan,
           'NANOGrav_15_yrs_SNR': np.nan,
           'LISA_SNR': np.nan,
           'B-DECIGO_SNR': np.nan,
           'DECIGO_SNR': np.nan,
           'BBO_SNR': np.nan,
           'ET_SNR': np.nan,
           'muAres_SNR': np.nan,
           'HLV_O2_SNR': np.nan,
           'HLVK_design_SNR': np.nan,
           'f_peak_Hz': np.nan,
           'h2OmegaGW_peak': np.nan,
           'f_pivot_Hz': np.nan,
           'h2OmegaGW_at_pivot': np.nan,
           'DNeff_GW': np.nan}

    return res
