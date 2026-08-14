# Local DCMFNet workflow

The Thesis repository owns synthetic data generation, local training, evaluation, and export. The separate Clinical Risk AI Agent repository owns inference only. Runtime inference does not import this repository or read a dataset.

From the Thesis repository:

```bash
python generate_synthetic_data.py --n-samples 20000 --seed 42
python train.py
python export_model.py
```

Training writes `artifacts/dcmfnet_pos.pt` and `artifacts/dcmfnet_neg.pt`, each with its local audit JSON. Export writes `exports/dcmfnet_pos.pt` and `exports/dcmfnet_neg.pt`, each with its corresponding `.metadata.json` file. Use `--target Pos` or `--target Neg` only when a single export is needed.

The training command uses grouped `cmpair` splits, fits preprocessing on training rows only, evaluates the held-out test split, and embeds the feature schema in the artifact. The audit JSON is for local evaluation and is not needed by serving.

Copy only the exported artifact and metadata to the Clinical Risk AI Agent repository:

```bash
cp exports/dcmfnet_pos.pt ../Clinical-Risk-AI-Agent/model_artifacts/
cp exports/dcmfnet_pos.metadata.json ../Clinical-Risk-AI-Agent/model_artifacts/
cp exports/dcmfnet_neg.pt ../Clinical-Risk-AI-Agent/model_artifacts/
cp exports/dcmfnet_neg.metadata.json ../Clinical-Risk-AI-Agent/model_artifacts/
cd ../Clinical-Risk-AI-Agent
python -m pip install '.[api]'
dcmfnet-api --artifact model_artifacts/dcmfnet_pos.pt --host 127.0.0.1 --port 8000
```

The FastAPI runtime loads one artifact at startup and exposes `/health`, `/v1/schema`, and `/v1/predict`. It requires only the inference runtime, the `.pt` artifact, and the schema metadata. It does not require pandas, scikit-learn, training code, synthetic CSVs, or the Thesis repository.

This is research decision-support software, not a diagnostic system or a validated clinical risk probability. Do not use its output as the sole basis for care decisions.
