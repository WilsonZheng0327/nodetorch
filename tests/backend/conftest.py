"""Shared backend test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_saved_runs(tmp_path, monkeypatch):
    """Keep test training runs out of the user's real storage/runs.

    The training tests exercise the real run_training(), which persists a run
    record via persistence.runs_store — so every `pytest` invocation was adding
    junk 1-epoch entries to the dashboard's Runs tab. save_run reads the
    module-global RUNS_DIR at call time, so pointing it at a temp dir is enough.
    """
    from persistence import runs_store

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(runs_store, "RUNS_DIR", runs_dir)
