# main.py — FastAPI backend for NodeTorch.
#
# Routes are split into APIRouter modules under backend/api/ (and the agent's
# self-contained slice in backend/agent/routes.py). This module wires them all
# together: app creation, CORS, logging, and router includes.

import warnings
warnings.filterwarnings("ignore", message="dtype.*align")

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import system, training, inference, generate, visualize, models, data, runs, library, export
from agent import routes as agent_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nodetorch")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
