"""Inference-only FastAPI application for one exported DCMFNet artifact."""

import argparse
from pathlib import Path
from typing import Any

from .predictor import DCMFNetPredictor, PredictionError


def create_app(artifact: str | Path, device: str = "cpu") -> Any:
    """Load one predictor at startup and bind inference-only routes."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install the API dependencies with `pip install .[api]`") from exc

    predictor = DCMFNetPredictor(artifact, device=device)

    class PredictRequest(BaseModel):
        records: list[dict[str, float]] = Field(min_length=1, max_length=256)

    app = FastAPI(
        title="DCMFNet Clinical Decision-Support API",
        version="1.0.0",
        description="Research symptom-severity inference; not a diagnostic service.",
    )
    app.state.predictor = predictor

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "target": predictor.metadata["target"]}

    @app.get("/v1/schema")
    def schema() -> dict[str, Any]:
        return predictor.schema_response()

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            return predictor.predict(request.records)
        except PredictionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install the API dependencies with `pip install .[api]`") from exc
    uvicorn.run(create_app(args.artifact, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
