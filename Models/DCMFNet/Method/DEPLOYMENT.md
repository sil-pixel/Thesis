# Local DCMFNet workflow

The Thesis repository owns synthetic data generation, local training, evaluation, and export. The separate Clinical Risk AI Agent repository owns inference only. Runtime inference does not import this repository or read a dataset.

From the Thesis repository:

```bash
python generate_synthetic_data.py --n-samples 20000 --seed 42
python train.py
python export_model.py
```

Outputs are `data/synthetic_dcmfnet.csv`, `artifacts/dcmfnet_pos.pt` plus its local audit JSON, and `exports/dcmfnet_pos.pt` plus `exports/dcmfnet_pos.metadata.json`. Train the negative target with `python train.py --target Neg --output artifacts/dcmfnet_neg.pt` and export it with matching `--checkpoint` and `--output` arguments.

The training command uses grouped `cmpair` splits, fits preprocessing on training rows only, evaluates the held-out test split, and embeds the feature schema in the artifact. The audit JSON is for local evaluation and is not needed by serving.

Copy only the exported artifact and metadata to the Clinical Risk AI Agent repository:

```bash
cp exports/dcmfnet_pos.pt ../Clinical-Risk-AI-Agent/model_artifacts/
cp exports/dcmfnet_pos.metadata.json ../Clinical-Risk-AI-Agent/model_artifacts/
cd ../Clinical-Risk-AI-Agent
python -m pip install '.[api]'
dcmfnet-api --artifact model_artifacts/dcmfnet_pos.pt --host 127.0.0.1 --port 8000
```

The FastAPI runtime loads one artifact at startup and exposes `/health`, `/v1/schema`, and `/v1/predict`. It requires only the inference runtime, the `.pt` artifact, and the schema metadata. It does not require pandas, scikit-learn, training code, synthetic CSVs, or the Thesis repository.

This is research decision-support software, not a diagnostic system or a validated clinical risk probability. Do not use its output as the sole basis for care decisions.
