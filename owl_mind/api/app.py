"""FastAPI entry. Routes are added by the plans that first use them."""

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Importing telemetry registers the llm_* metrics.
from owl_mind.core import telemetry  # noqa: F401

app = FastAPI(title="Owl-Mind Harness")


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
