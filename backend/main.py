# main.py — FastAPI backend for NodeTorch.
#
# Routes are split into APIRouter modules under backend/api/ (and the agent's
# self-contained slice in backend/agent/routes.py). This module wires them all
# together: app creation, CORS, logging, and router includes.

import warnings
warnings.filterwarnings("ignore", message="dtype.*align")

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import system, training, inference, generate, visualize, models, data, runs, library, export
from agent import routes as agent_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nodetorch")
# Quiet httpx's per-request "200 OK" lines (noise); keep openai's retry/backoff
# logs (a separate "openai" logger) and our own "nodetorch.agent" logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
# AGENT_DEBUG=1 dumps the full agent request (system + messages + tools) per turn.
if os.environ.get("AGENT_DEBUG"):
    logging.getLogger("nodetorch.agent").setLevel(logging.DEBUG)

app = FastAPI(title="NodeTorch Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(generate.router)
app.include_router(visualize.router)
app.include_router(models.router)
app.include_router(data.router)
app.include_router(runs.router)
app.include_router(library.router)
app.include_router(export.router)
app.include_router(agent_routes.router)


if __name__ == "__main__":
    import uvicorn

    # NODETORCH_DEV=1 enables hot reload — the server restarts on Python changes
    # so you don't have to stop/start it by hand while developing. reload needs
    # the app as an import string (not the app object); the reloader spawns a
    # child process that imports it, so we put this dir on PYTHONPATH (inherited
    # by the child) to make "main" importable no matter which directory you
    # launched from. Default (unset) runs the app object directly — no watcher.
    if os.environ.get("NODETORCH_DEV") == "1":
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["PYTHONPATH"] = backend_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[backend_dir])
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
