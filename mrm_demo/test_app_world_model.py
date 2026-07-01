"""Regression checks for selecting the demo world-model implementation."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrm_demo import app


def _scenario(scenario_id: str = "S4"):
    return app.find_scenario(app.load_scenarios(app._scenario_path()), scenario_id)


def test_run_pipeline_selects_world_model() -> None:
    state = _scenario()

    mock_result = app.run_pipeline(state, model_kind="mock")
    assert mock_result["world_model_kind"] == "mock"
    assert "method_note" not in mock_result["world_model"]

    paper_result = app.run_pipeline(state, model_kind="paper")
    assert paper_result["world_model_kind"] == "paper"
    assert paper_result["world_model"]["method_note"] == "lightweight_dawm_mpc_reproduction"


def test_cli_world_model_argument_defaults_to_mock() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = app.run_cli(["--scenario", "S4", "--json"])

    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["world_model_kind"] == "mock"


def test_cli_world_model_argument_accepts_paper() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = app.run_cli(["--scenario", "S4", "--world-model", "paper", "--json"])

    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["world_model_kind"] == "paper"
    assert payload["world_model"]["method_note"] == "lightweight_dawm_mpc_reproduction"


if __name__ == "__main__":
    test_run_pipeline_selects_world_model()
    test_cli_world_model_argument_defaults_to_mock()
    test_cli_world_model_argument_accepts_paper()
    print("All app world-model tests passed.")
