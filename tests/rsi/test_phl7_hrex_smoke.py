"""Bounded HREX checks reject incomplete trajectories and exchange schedules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.run_phl7_hrex_smoke import EXCHANGE, STEPS, allocation_plan, exchange_steps, parse_series, short_mdp


def test_short_protocol_changes_only_declared_duration_output_and_seed_fields():
    source = "integrator = md\nnsteps = 250000000\ndt = .002\nref_t = 300 300\ngen_vel = no\nconstraints = h-bonds\n"
    text, updates = short_mdp(source, 2)
    assert updates["nsteps"] == "2000" and updates["ld-seed"] == "43003"
    assert all(
        line in text
        for line in ["integrator = md", "dt = .002", "ref_t = 300 300", "gen_vel = no", "constraints = h-bonds"]
    )
    assert "250000000" not in text and text.count("nsteps") == 1


def xvg(path, mutation=None):
    head = "".join(
        f'@ s{i} legend "{name}"\n' for i, name in enumerate(["Potential", "Temperature", "Pressure", "Density"])
    )
    rows = [[i * 0.2, -549000, 300, 1, 1000] for i in range(21)]
    if mutation == "missing_frame":
        rows.pop()
    elif mutation == "wrong_duration":
        rows[-1][0] = 3.9
    elif mutation == "nan":
        rows[2][1] = float("nan")
    elif mutation == "temperature":
        rows[5][2] = 600
    elif mutation == "density":
        rows[7][4] = 2000
    path.write_text(head + "".join(" ".join(map(str, row)) + "\n" for row in rows))
    return path


def test_finite_four_ps_observables_are_summarized(tmp_path):
    result = parse_series(xvg(tmp_path / "e.xvg"))
    assert result["Temperature"] == {"min": 300, "max": 300, "mean": 300}


@pytest.mark.parametrize("mutation", ["missing_frame", "wrong_duration", "nan", "temperature", "density"])
def test_incomplete_or_invalid_dynamics_cannot_pass(tmp_path, mutation):
    with pytest.raises(ValueError):
        parse_series(xvg(tmp_path / "e.xvg", mutation))


def log(steps):
    return "".join(f"Replica exchange at step {step} time {step * 0.002}\n" for step in steps)


def test_exchange_schedule_excludes_initial_and_final_steps():
    expected = list(range(EXCHANGE, STEPS, EXCHANGE))
    assert exchange_steps(log(expected)) == expected and len(expected) == 19


@pytest.mark.parametrize("mutation", ["missing_exchange", "duplicate", "lincs"])
def test_incomplete_exchange_or_constraint_warning_is_rejected(mutation):
    steps = list(range(EXCHANGE, STEPS, EXCHANGE))
    if mutation == "missing_exchange":
        steps.pop()
    elif mutation == "duplicate":
        steps.insert(0, 100)
    text = log(steps) + ("LINCS WARNING" if mutation == "lincs" else "")
    with pytest.raises(ValueError):
        exchange_steps(text)


def test_cpu_admission_respects_affinity_and_stricter_parent_quota():
    plan = allocation_plan([4, 5, 6, 7, 9, 132, 133, 134, 135], [])
    assert plan["requested_threads"] == 8 and plan["effective_logical_cpu_budget"] == 9
    assert allocation_plan(range(256), [20, 8.9])["effective_logical_cpu_budget"] == 8
    with pytest.raises(ValueError, match="exceed"):
        allocation_plan(range(256), [20, 7.9])
    with pytest.raises(ValueError, match="exceed"):
        allocation_plan(range(7), [])


@pytest.mark.parametrize("affinity,quotas", [([], []), (range(9), [float("nan")]), (range(9), [0])])
def test_cpu_admission_rejects_absent_or_invalid_capacity(affinity, quotas):
    with pytest.raises(ValueError, match="Invalid"):
        allocation_plan(affinity, quotas)
