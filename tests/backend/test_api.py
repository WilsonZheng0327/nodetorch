"""API surface tests — verify the router split preserves the full route set
and a few representative endpoints behave as expected."""

from fastapi.testclient import TestClient

import main


# The full set of registered route paths after the APIRouter split,
# built from the endpoint → module mapping.
EXPECTED_PATHS = {
    # system
    "/health",
    "/system-info",
    "/set-device",
    # training
    "/train",
    "/ws",
    # inference
    "/infer",
    "/evaluate-test",
    # generate
    "/gan-generate",
    "/denoise-step-through",
    "/latent-grid",
    "/generate-text",
    # visualize
    "/step-through",
    "/backward-step-through",
    "/simulate-backprop",
    "/loss-landscape",
    "/activation-max",
    "/layer-detail",
    # models
    "/save-model",
    "/load-model",
    "/saved-models",
    "/download-weights",
    "/upload-weights",
    "/download-model",
    "/upload-model",
    # data
    "/dataset/{dataset_type:path}",
    "/tokenizer/preview",
    "/augmentation-preview",
    # runs
    "/runs",
    "/runs/{run_id}",       # GET + DELETE share the same path
    # library
    "/presets",
    "/presets/load",
    "/blocks",
    "/blocks/save",
    "/blocks/{filename:path}",  # GET + DELETE share the same path
    # export
    "/export-python",
    # agent
    "/agent/providers",
    "/agent/config",        # GET + POST share the same path
    "/agent/test",
    "/agent",
}


def _app_route_paths():
    """All registered route paths on the app (excluding FastAPI's built-in
    docs/openapi routes)."""
    builtin = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    return {
        r.path
        for r in main.app.routes
        if getattr(r, "path", None) and r.path not in builtin
    }


def test_registered_routes_exactly_match_expected():
    paths = _app_route_paths()
    missing = EXPECTED_PATHS - paths
    extra = paths - EXPECTED_PATHS
    assert not missing, f"Missing routes: {sorted(missing)}"
    assert not extra, f"Unexpected routes: {sorted(extra)}"
    assert paths == EXPECTED_PATHS


def test_route_count():
    """Total endpoint count — counts each method on shared paths separately."""
    builtin = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    http_endpoints = 0
    ws_endpoints = 0
    for r in main.app.routes:
        path = getattr(r, "path", None)
        if not path or path in builtin:
            continue
        methods = getattr(r, "methods", None)
        if methods:
            # exclude auto-added HEAD/OPTIONS — count GET/POST/DELETE/etc.
            http_endpoints += len(methods - {"HEAD", "OPTIONS"})
        else:
            ws_endpoints += 1
    assert http_endpoints + ws_endpoints == 42


def test_health():
    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_agent_providers():
    client = TestClient(main.app)
    resp = client.get("/agent/providers")
    assert resp.status_code == 200
    data = resp.json()
    provider_names = {p.get("name") for p in data["providers"]}
    assert "openai_compat" in provider_names
    assert "anthropic" in provider_names


def test_agent_config_never_exposes_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.config._CONFIG_PATH", tmp_path / "cfg.json")
    client = TestClient(main.app)
    resp = client.get("/agent/config")
    assert resp.status_code == 200
    assert "api_key" not in resp.json()
