from __future__ import annotations

from pathlib import Path

import pytest

from cqedtoolbox.protocols.operations.single_qubit import pi_spec


class MutableParameter:
    def __init__(self, value):
        self.value = value

    def __call__(self, value=None):
        if value is None:
            return self.value
        self.value = value
        return value


def make_operation():
    operation = object.__new__(pi_spec.PiSpectroscopy)
    operation.name = "PiSpectroscopy"
    operation.qubit_gain = MutableParameter(0.2)
    operation.pi_length = MutableParameter(0.1)
    return operation


def test_qick_pi_spec_temporarily_scales_and_restores_pulse(monkeypatch):
    operation = make_operation()
    monkeypatch.setattr(pi_spec, "PiSpecProgram", lambda: object())

    def run(sweep, dataset_name, operation_name):
        assert operation.qubit_gain() == pytest.approx(0.02)
        assert operation.pi_length() == pytest.approx(1.0)
        return Path("data"), object()

    monkeypatch.setattr(pi_spec, "run_and_save_sweep", run)

    assert operation._PI_SPEC_PULSE_FACTOR == 10
    assert operation._measure_qick() == Path("data")
    assert operation.qubit_gain() == pytest.approx(0.2)
    assert operation.pi_length() == pytest.approx(0.1)


def test_qick_pi_spec_restores_pulse_after_measurement_error(monkeypatch):
    operation = make_operation()
    monkeypatch.setattr(pi_spec, "PiSpecProgram", lambda: object())

    def fail(*args, **kwargs):
        raise RuntimeError("measurement failed")

    monkeypatch.setattr(pi_spec, "run_and_save_sweep", fail)

    with pytest.raises(RuntimeError, match="measurement failed"):
        operation._measure_qick()

    assert operation.qubit_gain() == pytest.approx(0.2)
    assert operation.pi_length() == pytest.approx(0.1)
