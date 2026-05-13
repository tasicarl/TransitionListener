"""CSV output shaping utilities for TransitionListener interfaces.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Column groups with preferred ordering
PTA_LABEL_ORDER = ["NG15_14bins", "NG15_5bins", "NG12_5bins", "IPTA2_13bins"]
PTA_COLUMNS_ORDER = [
    item
    for label in PTA_LABEL_ORDER
    for item in (
        f"lnL_smoothened_{label}",
        f"lnL_PTArcade_{label}",
        f"lnL_mock_{label}",
    )
]
BASE_RESULT_COLUMNS = ["smoothened_lnL", "PTArcade_lnL", "mock_lnL", "error", "errormsg"]
DERIVED_PREFERRED_ORDER = [
    "alpha",
    "alpha_theta",
    "alpha_thetabar",
    "alpha_inf",
    "alpha_eq",
    "betaH_S3",
    "betaH_RH",
    "RH",
    "Treh_SM_GeV",
    "Tperc_SM_GeV",
    "g_eff_tot_reh",
    "h_eff_tot_reh",
    "kappa_phi",
    "kappa_sw",
    "kappa_turb",
    "g0",
    "h0",
    "v_wall",
    "D",
    "c_s",
    "c_s_sym",
    "c_s_bro",
    "step",
    "total_steps",
    "Tnuc_SM_GeV",
    "Tcrit_SM_GeV",
    "Tf_SM_GeV",
    "xi_crit",
]
WARNING_COLUMN_ORDER = [
    "WARNING:too_weak_to_compute_perc",
    "WARNING:no_perc_splines",
    "WARNING:betaH_small",
    "WARNING:betaH_very_small",
    "WARNING:betaH_mismatch",
    "WARNING:betaH_nonfinite",
    "WARNING:nucleationRate_nonexponential",
    "WARNING:not_T0_global_min",
]
SNR_COLUMN_ORDER = [
    "B-DECIGO_SNR",
    "BBO_SNR",
    "DECIGO_SNR",
    "EPTA_18_yrs_SNR",
    "ET_SNR",
    "HLVK_design_SNR",
    "HLV_O2_SNR",
    "LISA_SNR",
    "NANOGrav_11_yrs_SNR",
    "NANOGrav_15_yrs_SNR",
    "SKA_10_yrs_SNR",
    "SKA_20_yrs_SNR",
    "SKA_5_yrs_SNR",
    "muAres_SNR",
]
PEAK_COLUMN_PRIORITY = [
    "f_peak_Hz",
    "h2OmegaGW_peak",
    "f_pivot_Hz",
    "h2OmegaGW_at_pivot",
    "DNeff_GW",
]
STATIC_ORDER_GROUPS = (
    WARNING_COLUMN_ORDER,
    PTA_COLUMNS_ORDER,
    SNR_COLUMN_ORDER,
    PEAK_COLUMN_PRIORITY,
)


class OutputRandomscan:
    """Ouput handler for the randomscan.

    It opens a file and appends new results as a row to it.
    This is inteded to be used with the listener/worker threads in the randomscan
    module.
    """

    def __init__(self, output_file: str):
        """Open ``output_file`` in append mode and keep the handle alive across writes."""
        self.file = output_file

        try:
            self.f = open(self.file, 'a')
        except Exception as e:
            print(e)

    def stop(self):
        """Stop the output handler and close the file.

        Parameters
        ----------
        None.

        Returns
        -------
        None."""
        self.f.close()

    def write_row(self, inp_params: dict, results: dict):
        """Write a data point to the output file.

        Parameters
        ----------
        inp_params : dict
            The input parameters of the potential.
        results : dict
            The results that should be written to the csv file.

        Returns
        -------
        None.
        """
        data = inp_params.copy()

        if "_meta_mass_columns" in results.keys():
            results.pop("_meta_mass_columns")

        data.update(results)
        if os.path.getsize(self.file) == 0:
            # write the header:
            header = ','.join(list(data.keys()))
            self.f.write(header + '\n')

        row = ','.join([str(val) for val in data.values()])
        self.f.write(row + '\n')
        self.f.flush()


class OutputSchema:
    """Manage the CSV schema used for sampler outputs."""

    def __init__(self, potential: object | None, derived_param_names: Iterable[str]):
        """Store the optional potential factory and the preferred derived-column order."""
        self.potential = potential
        self.derived_keys = list(derived_param_names)

    def append_row(self, output_file: str, inputparams_dict: dict, res_params: dict) -> None:
        """Append a single result row to ``output_file`` using the configured schema."""
        row, column_order = self._prepare_row(inputparams_dict, res_params)

        file_exists = os.path.exists(output_file)
        file_nonempty = file_exists and os.path.getsize(output_file) > 0

        if file_nonempty:
            existing_df = pd.read_csv(output_file)
            total_order = list(dict.fromkeys(column_order + list(existing_df.columns)))
            row_data = {col: row.get(col, np.nan) for col in total_order}
            row_df = pd.DataFrame([row_data], columns=total_order)
            existing_df = existing_df.reindex(columns=total_order, fill_value=np.nan)
            combined = pd.concat([existing_df, row_df], ignore_index=True)
            combined.to_csv(output_file, index=False)
            return

        row_data = {col: row.get(col, np.nan) for col in column_order}
        row_df = pd.DataFrame([row_data], columns=column_order)
        row_df.to_csv(output_file, index=False)

    def _prepare_row(self, inputparams_dict: dict, res_params: dict) -> tuple[dict, list[str]]:
        """Merge input/output data and return the normalised row plus its column order."""
        row = {**inputparams_dict}
        row.update(res_params)

        mass_columns = list(row.pop("_meta_mass_columns", []))
        row.pop("lnL", None)

        meta_keys = [key for key in row if key.startswith("_meta_")]
        for key in meta_keys:
            row.pop(key, None)

        error_value = row.get("error")
        if self.potential is not None and error_value not in (0, None):
            for column, value in self._compute_mass_columns(inputparams_dict):
                if column not in mass_columns:
                    mass_columns.append(column)
                row[column] = value

        column_order = _build_column_order(
            inputparams_dict,
            self.derived_keys,
            row.keys(),
            mass_columns,
        )
        return row, column_order

    def _compute_mass_columns(self, inputparams_dict: dict) -> list[tuple[str, float]]:
        """Compute fallback zero-temperature mass columns for failed parameter points."""
        if self.potential is None:
            return []
        try:
            potential = self.potential(inputparams_dict, verbose=False)  # type: ignore[attr-defined]
        except Exception:
            return []

        getter = getattr(potential, "get_zero_temperature_mass_spectrum", None)
        if getter is None:
            return []

        try:
            entries = getter()
        except Exception:
            entries = []

        used_columns: set[str] = set()
        results: list[tuple[str, float]] = []

        def _unique_column(base: str) -> str:
            candidate = base or "mass"
            index = 2
            while candidate in used_columns:
                candidate = f"{base}_{index}"
                index += 1
            used_columns.add(candidate)
            return candidate

        for entry in entries or []:
            text_label = str(entry.get("text", "")).strip()
            kind = entry.get("kind", "boson")
            index = int(entry.get("index", 0))
            base_name = text_label or f"mass_spectrum_T0_{kind}_{index:02d}"
            column = _unique_column(base_name)

            mass_value = entry.get("mass_GeV", np.nan)
            if isinstance(mass_value, (np.ndarray, list, tuple)):
                mass_value = np.asarray(mass_value).flatten()[0] if np.asarray(mass_value).size else np.nan
            try:
                value = float(mass_value)
            except (TypeError, ValueError):
                value = np.nan

            if not np.isfinite(value):
                if column.startswith("m_G"):
                    value = 0.0
                else:
                    value = np.nan

            results.append((column, value))

        return results


def _build_column_order(
    inputparams_dict: dict,
    derived_keys: Sequence[str],
    row_keys: Iterable[str],
    mass_columns: Sequence[str],
) -> list[str]:
    """Construct a stable CSV column order that keeps related observables grouped together."""
    base_order: list[str] = list(inputparams_dict.keys())
    base_order.extend(BASE_RESULT_COLUMNS)
    base_order.extend(
        col for col in DERIVED_PREFERRED_ORDER if col in derived_keys or col in row_keys
    )
    base_order.extend(col for col in derived_keys if col not in DERIVED_PREFERRED_ORDER)
    for group in STATIC_ORDER_GROUPS:
        base_order.extend(group)

    remaining = sorted(
        col
        for col in row_keys
        if col not in base_order and col not in mass_columns
    )
    base_order.extend(remaining)
    base_order.extend(mass_columns)
    return list(dict.fromkeys(base_order))
