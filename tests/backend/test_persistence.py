"""Tests for persistence/runs_store.py — the training-run store behind the
dashboard's Runs tab. The conftest autouse fixture already points RUNS_DIR at a
temp dir, so these exercise the real file I/O in isolation.
"""
from persistence import runs_store


def _record(**over):
    rec = {
        "timestamp": "2026-07-14T10:00:00",
        "datasetType": "data.mnist",
        "epochs": 3,
        "finalLoss": 0.5,
        "finalAccuracy": 0.9,
        "epochHistory": [{"epoch": 1, "loss": 0.5}],
    }
    rec.update(over)
    return rec


class TestSaveLoad:
    def test_round_trips_the_full_record_including_epoch_history(self):
        run_id = runs_store.save_run(_record())
        loaded = runs_store.load_run(run_id)
        assert loaded is not None
        assert loaded["datasetType"] == "data.mnist"
        assert loaded["epochHistory"] == [{"epoch": 1, "loss": 0.5}]

    def test_generates_an_id_when_none_supplied(self):
        run_id = runs_store.save_run(_record())
        assert run_id and run_id.startswith("run-")

    def test_respects_a_caller_supplied_id(self):
        run_id = runs_store.save_run(_record(id="my-run"))
        assert run_id == "my-run"
        assert runs_store.load_run("my-run") is not None

    def test_load_unknown_id_returns_none(self):
        assert runs_store.load_run("does-not-exist") is None


class TestListRuns:
    def test_returns_summary_fields_only_not_epoch_history(self):
        runs_store.save_run(_record(id="r1"))
        listed = runs_store.list_runs()
        assert len(listed) == 1
        assert listed[0]["id"] == "r1"
        assert "epochHistory" not in listed[0]  # summary view is trimmed

    def test_sorted_newest_first_by_timestamp(self):
        runs_store.save_run(_record(id="old", timestamp="2026-07-01T00:00:00"))
        runs_store.save_run(_record(id="new", timestamp="2026-07-14T00:00:00"))
        ids = [r["id"] for r in runs_store.list_runs()]
        assert ids == ["new", "old"]

    def test_empty_store_lists_nothing(self):
        assert runs_store.list_runs() == []


class TestDelete:
    def test_delete_removes_the_run_and_reports_success(self):
        runs_store.save_run(_record(id="doomed"))
        assert runs_store.delete_run("doomed") is True
        assert runs_store.load_run("doomed") is None

    def test_deleting_a_missing_run_returns_false(self):
        assert runs_store.delete_run("ghost") is False


class TestBuildRunRecord:
    def _args(self, epoch_results):
        return dict(
            graph_data={"graph": {"nodes": [{"id": "a"}, {"id": "b"}]}},
            epoch_results=epoch_results,
            optimizer_props={"__type__": "adam", "epochs": 2, "lr": 0.001},
            data_node={"type": "data.mnist", "properties": {"batchSize": 64}},
            duration_seconds=12.34,
            module_param_count=1000,
        )

    def test_summarizes_final_and_best_metrics_from_epoch_history(self):
        epochs = [
            {"epoch": 1, "loss": 0.8, "accuracy": 0.6, "valAccuracy": 0.55},
            {"epoch": 2, "loss": 0.4, "accuracy": 0.9, "valAccuracy": 0.85},
        ]
        rec = runs_store.build_run_record(**self._args(epochs))
        assert rec["finalLoss"] == 0.4
        assert rec["finalAccuracy"] == 0.9
        assert rec["bestValAccuracy"] == 0.85
        assert rec["optimizer"] == "adam"
        assert rec["nodeCount"] == 2
        assert rec["totalParams"] == 1000
        assert rec["duration"] == 12.3  # rounded to 1 dp

    def test_best_val_accuracy_is_the_max_not_the_last(self):
        epochs = [
            {"epoch": 1, "valAccuracy": 0.9},
            {"epoch": 2, "valAccuracy": 0.7},  # regressed
        ]
        rec = runs_store.build_run_record(**self._args(epochs))
        assert rec["bestValAccuracy"] == 0.9
        assert rec["finalValAccuracy"] == 0.7

    def test_trims_epoch_history_to_charting_essentials(self):
        epochs = [{"epoch": 1, "loss": 0.5, "heavyField": [1, 2, 3], "accuracy": 0.6}]
        rec = runs_store.build_run_record(**self._args(epochs))
        assert rec["epochHistory"] == [
            {
                "epoch": 1,
                "loss": 0.5,
                "accuracy": 0.6,
                "valLoss": None,
                "valAccuracy": None,
                "learningRate": None,
                "time": None,
            }
        ]

    def test_record_survives_a_save_load_round_trip(self):
        rec = runs_store.build_run_record(**self._args([{"epoch": 1, "loss": 0.5}]))
        run_id = runs_store.save_run(rec)
        assert runs_store.load_run(run_id)["totalParams"] == 1000
