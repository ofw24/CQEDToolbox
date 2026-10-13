"""Qubit-temperature measurement from e-f time-Rabi oscillation amplitudes."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, k

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Cosine
from labcore.data.datadict_storage import load_as_xr
from labcore.protocols.base import (
    CheckResult,
    EvaluateResult,
    ProtocolOperation,
    serialize_fit_params,
)
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import (
    measure_ef_time_rabi,
    measure_ef_time_rabi_after_ge_pi,
    measure_time_rabi,
)
from cqedtoolbox.protocols.parameters import (
    EFDriveGain,
    QubitFrequency,
    QubitTemperature,
    QubitTemperatureError,
    Repetition,
)
from cqedtoolbox.readout.qubit_readout import rotate_complex_qubit_data


logger = logging.getLogger(__name__)


class QubitTemperatureOperation(ProtocolOperation):
    """Infer the effective qubit temperature from prepared and thermal e-f Rabi traces.

    The estimator assumes negligible thermal population in ``|f>``.  With the
    prepared trace amplitude ``A_ref`` and unprepared amplitude ``A_sig``, it
    uses ``P_e = A_sig / (A_sig + A_ref)`` and the two-level Boltzmann relation.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        self._register_inputs(
            repetitions=Repetition(params),
            qubit_frequency=QubitFrequency(params),
            ef_drive_gain=EFDriveGain(params),
        )
        self._register_outputs(
            temperature=QubitTemperature(params),
            temperature_error=QubitTemperatureError(params),
        )
        self._register_check("temperature_fit", self._check_temperature_fit)
        self._register_success_update(self.temperature, lambda: self.temperature_value)
        self._register_success_update(self.temperature_error, lambda: self.temperature_error_value)

        self.normal_time_rabi_data_loc: Path | None = None
        self.reference_data_loc: Path | None = None
        self.signal_data_loc: Path | None = None
        self.independents = {"times": []}
        self.dependents = {"reference_signal": []}
        self.reference_signal: np.ndarray | None = None
        self.signal_signal: np.ndarray | None = None
        self.reference_fit = None
        self.signal_fit = None
        self.reference_snr: float | None = None
        self.signal_snr: float | None = None
        self.pe: float | None = None
        self.pe_error: float | None = None
        self.temperature_value: float | None = None
        self.temperature_error_value: float | None = None
        self.fit_error: str | None = None

    def _measure_opx(self) -> Path:
        logger.info("Starting g-e time Rabi calibration trace")
        self.normal_time_rabi_data_loc = measure_time_rabi()
        logger.info("Starting prepared e-f time Rabi reference trace")
        self.reference_data_loc = measure_ef_time_rabi_after_ge_pi()
        logger.info("Starting thermal e-f time Rabi signal trace")
        self.signal_data_loc = measure_ef_time_rabi()
        return self.reference_data_loc

    def _measure_qick(self) -> Path:
        raise NotImplementedError("Qubit-temperature measurement is currently implemented for OPX only")

    def _measure_dummy(self) -> Path:
        raise NotImplementedError("Qubit-temperature measurement has no dummy backend")

    @staticmethod
    def _load_rotated_trace(location: Path) -> tuple[np.ndarray, np.ndarray]:
        data = load_as_xr(location)
        if "repetition" in data.dims:
            data = data.mean("repetition")
        rotated, _ = rotate_complex_qubit_data(data)
        return rotated["pulse_len"].values, rotated["signal"].values

    def _load_data_opx(self) -> None:
        if self.reference_data_loc is None or self.signal_data_loc is None:
            raise RuntimeError("Both e-f time-Rabi traces must be acquired before loading data")
        reference_times, self.reference_signal = self._load_rotated_trace(self.reference_data_loc)
        signal_times, self.signal_signal = self._load_rotated_trace(self.signal_data_loc)
        if not np.array_equal(reference_times, signal_times):
            raise RuntimeError("Prepared and thermal e-f Rabi traces have different time samples")
        self.independents["times"] = reference_times
        self.dependents["reference_signal"] = self.reference_signal

    def _load_data_qick(self) -> None:
        raise NotImplementedError("Qubit-temperature measurement is currently implemented for OPX only")

    def _load_data_dummy(self) -> None:
        raise NotImplementedError("Qubit-temperature measurement has no dummy backend")

    @staticmethod
    def _fit_cosine(times: np.ndarray, signal: np.ndarray):
        fit = Cosine(times, signal)
        result = fit.run(fit)
        curve = result.eval()
        residuals = signal - curve
        amplitude = abs(result.params["A"].value)
        noise = np.std(residuals)
        snr = np.inf if noise == 0 else amplitude / (4 * noise)
        return result, curve, snr

    def _analyze_default(self) -> None:
        if self.reference_signal is None or self.signal_signal is None:
            raise RuntimeError("Load both e-f time-Rabi traces before analysis")

        times = np.asarray(self.independents["times"])
        self.fit_error = None
        try:
            self.reference_fit, reference_curve, self.reference_snr = self._fit_cosine(times, self.reference_signal)
            self.signal_fit, signal_curve, self.signal_snr = self._fit_cosine(times, self.signal_signal)
            self._calculate_temperature()
        except Exception as error:
            self.fit_error = str(error)
            logger.exception("Unable to fit e-f Rabi traces")
            reference_curve = np.full_like(times, np.nan, dtype=float)
            signal_curve = np.full_like(times, np.nan, dtype=float)

        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ds.add(
                normal_time_rabi_data_loc=str(self.normal_time_rabi_data_loc),
                reference_data_loc=str(self.reference_data_loc),
                signal_data_loc=str(self.signal_data_loc),
                reference_fit=(serialize_fit_params(self.reference_fit.params)
                               if self.reference_fit is not None else None),
                signal_fit=(serialize_fit_params(self.signal_fit.params)
                            if self.signal_fit is not None else None),
                pe=self.pe,
                pe_error=self.pe_error,
                temperature=self.temperature_value,
                temperature_error=self.temperature_error_value,
                fit_error=self.fit_error,
            )

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
            for axis, signal, curve, label in (
                (axes[0], self.reference_signal, reference_curve, "Prepared e-f Rabi"),
                (axes[1], self.signal_signal, signal_curve, "Thermal e-f Rabi"),
            ):
                axis.plot(times, signal, ".", label="data")
                axis.plot(times, curve, label="cosine fit")
                axis.set_title(label)
                axis.set_xlabel("Pulse length (ns)")
                axis.set_ylabel("Rotated signal (a.u.)")
                axis.legend()
            fig.tight_layout()
            ds.add_figure(self.name, fig=fig)
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def analyze(self) -> None:
        self._analyze_default()

    def _calculate_temperature(self) -> None:
        assert self.reference_fit is not None and self.signal_fit is not None
        a_ref = abs(self.reference_fit.params["A"].value)
        a_sig = abs(self.signal_fit.params["A"].value)
        sigma_ref = self.reference_fit.params["A"].stderr
        sigma_sig = self.signal_fit.params["A"].stderr
        if sigma_ref is None or sigma_sig is None:
            raise ValueError("The Rabi-amplitude fit uncertainty is unavailable")
        if a_ref <= 0 or a_sig <= 0:
            raise ValueError("Both e-f Rabi amplitudes must be positive")

        self.pe = a_sig / (a_sig + a_ref)
        if not 0 < self.pe < 0.5:
            raise ValueError(f"Inferred P_e={self.pe:.6g} is outside the physical two-level range (0, 0.5)")
        denominator = (a_sig + a_ref) ** 2
        self.pe_error = np.sqrt(
            (a_ref * sigma_sig / denominator) ** 2
            + (a_sig * sigma_ref / denominator) ** 2
        )
        log_ratio = np.log((1 - self.pe) / self.pe)
        self.temperature_value = h * self.qubit_frequency() / (k * log_ratio)
        dtdpe = h * self.qubit_frequency() / (k * self.pe * (1 - self.pe) * log_ratio ** 2)
        self.temperature_error_value = abs(dtdpe) * self.pe_error

    def _check_temperature_fit(self) -> CheckResult:
        passed = (
            self.fit_error is None
            and self.temperature_value is not None
            and self.temperature_error_value is not None
            and np.isfinite(self.temperature_value)
            and np.isfinite(self.temperature_error_value)
        )
        if not passed:
            return CheckResult("temperature_fit", False, self.fit_error or "Temperature fit did not produce a finite result")
        return CheckResult(
            "temperature_fit",
            True,
            f"P_e={self.pe:.5g} ± {self.pe_error:.2g}; "
            f"T={self.temperature_value * 1e3:.3f} ± {self.temperature_error_value * 1e3:.3f} mK",
        )

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        figure = self.figure_paths[0] if self.figure_paths else None
        self.figure_paths.clear()
        self.report_output.append("## Qubit Temperature\n")
        self.report_output.append(f"Normal g-e time-Rabi data: `{self.normal_time_rabi_data_loc}`\n\n")
        self.report_output.append(f"Prepared e-f reference data: `{self.reference_data_loc}`\n\n")
        self.report_output.append(f"Thermal e-f signal data: `{self.signal_data_loc}`\n\n")
        if figure:
            self.report_output.append(figure)
        if self.reference_fit is not None and self.signal_fit is not None:
            self.report_output.append(
                "### e-f Rabi Fits\n"
                f"Reference amplitude: `{abs(self.reference_fit.params['A'].value):.6g}`\n\n"
                f"Signal amplitude: `{abs(self.signal_fit.params['A'].value):.6g}`\n\n"
            )
        return super().correct(result)
