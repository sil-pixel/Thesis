# DCMFNet training and clinical decision-support runtime

This directory contains a reproducible training command, a versioned model
artifact, a Python inference class, and an optional HTTP API. The deployed
output is a normalized symptom-severity estimate for either the positive or
negative SCZ18 target. It is not a calibrated probability of schizophrenia.

## Project structure

```text
Method/
├── dcmfnet/
│   ├── model.py              # DCMFNet architecture
│   ├── losses.py             # Imbalanced regression losses
│   ├── artifact.py           # Safe, inference-only artifact loader
│   ├── schema.py             # Runtime feature validation/preprocessing
│   ├── predictor.py          # Python inference interface
│   ├── api.py                # FastAPI application and server CLI
│   └── training/             # Offline-only data, engine, export, and CLI
├── legacy/                   # Original thesis workflows kept for reproducibility
├── tests/
├── Dockerfile.api
└── pyproject.toml
```

The top-level `api.py`, `model.py`, `loss.py`, `train.py`, `tuning.py`, and
training launchers are compatibility shims. New code should import from
`dcmfnet` and `dcmfnet.training`.

## Install

Use separate environments. In the confidential/offline training environment:

```bash
python -m pip install '.[train]'
```

In the API environment, install only inference and serving dependencies:

```bash
python -m pip install '.[api]'
```

The API extras do not install pandas or scikit-learn and the serving code never
imports `dcmfnet.training`.

## Train and export

The CSV must contain `cmpair`, `SCZ18_Pos_Norm` and/or `SCZ18_Neg_Norm`, and
features matching the modality prefixes declared in `dcmfnet/training/data.py`.
Feature order and fitted preprocessing parameters are embedded in the artifact
because they are required to transform a new record consistently. No source
rows, subject/twin IDs, labels, source pathname, or train/test frames are saved.

```bash
python offline_train_and_export.py \
  --data /secure/path/catss_final_data.csv \
  --target Pos \
  --output /secure/export/dcmfnet_pos.pt
```

The installed equivalent is `dcmfnet-train`.

Train `--target Neg` separately for the negative-symptom target. Training uses
grouped outer test and inner validation splits, so members of one `cmpair` do
not cross split boundaries. The `.audit.json` file written beside the artifact
contains training settings, row counts, and aggregate evaluation metrics for
confidential offline review. It is intentionally not embedded in the
deployable `.pt` artifact and must not be copied to the API environment. The
held-out test set is evaluated once and is not used for early stopping.

Do not start `api.py` or `dcmfnet-api` in the confidential training
environment. A virtual environment separates Python packages, but it does not
restrict filesystem or network access. Finish training, stop using that
environment, and move only the deployable `.pt` artifact to the serving
environment.

## Transfer the exported model through GitHub

The `.pt` artifact contains no patient rows or identifiers, but it does contain
learned weights and aggregate preprocessing parameters derived from the
training set. Treat it as a sensitive research artifact.

Use GitHub for transfer only when all of the following are true:

- The repository is private and its collaborator list has been reviewed.
- Storing the derived model on GitHub is permitted by the data-use agreement,
  ethics/IRB conditions, and institutional policy that apply to the project.
- The raw CSV, `.audit.json`, training logs, per-patient predictions, and direct
  identifiers are not committed.

If the repository is public, do not push the `.pt` file. Changing a repository
to private after a push does not retract copies that may already have been
cloned or forked.

The local `.gitignore` intentionally ignores `*.pt` and `*.audit.json`. After
confirming the conditions above, add only the exact deployable artifact:

```bash
mkdir -p deployable_models
cp /secure/export/dcmfnet_pos.pt deployable_models/dcmfnet_pos.pt

shasum -a 256 deployable_models/dcmfnet_pos.pt
git status --short
git add -f deployable_models/dcmfnet_pos.pt
git diff --cached --name-only
git commit -m "Add exported DCMFNet positive-symptom model"
git push origin HEAD
```

Before committing, `git diff --cached --name-only` must list only the intended
`.pt` model and any deliberately updated code or documentation. It must not list
the `.audit.json`, a CSV, or another confidential output. Record the SHA-256
value securely so the serving machine can verify the downloaded artifact.

On the separate serving machine, clone or pull the private repository into an
environment that contains no training or testing datasets, and verify the
artifact before starting the API:

```bash
git clone git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git
cd YOUR_PRIVATE_REPOSITORY/Models/DCMFNet/Method

shasum -a 256 ../../../deployable_models/dcmfnet_pos.pt
python -m venv api-env
source api-env/bin/activate
python -m pip install '.[api]'

dcmfnet-api \
  --artifact ../../../deployable_models/dcmfnet_pos.pt \
  --host 127.0.0.1 \
  --port 8000
```

Compare the SHA-256 output with the value recorded in the training environment.
The API loads this artifact once when the process starts and does not need or
attempt to locate the original dataset.

## Python/agent integration

An agent tool can keep one predictor in memory and pass a dictionary containing
all named features for each record:

```python
from dcmfnet import DCMFNetPredictor

predictor = DCMFNetPredictor("artifacts/dcmfnet_pos.pt")
required_input = predictor.schema_response()
result = predictor.predict([patient_feature_dictionary])
```

Missing, non-numeric, and infinite inputs are rejected. Python callers may use
NaN for a present-but-missing value, which is imputed with the training median;
HTTP callers must send numeric JSON values. Every key must still be present, so
accidental schema drift is visible to the caller.

## HTTP API

```bash
dcmfnet-api --artifact /path/to/dcmfnet_pos.pt --host 127.0.0.1 --port 8000
```

`create_app` loads the artifact once during process startup. Every request uses
that same in-memory predictor; prediction endpoints contain no training call
and perform no dataset reads.

Inspect `GET /v1/schema`, then send one or more complete feature records:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"records":[{"SUD15_example":0.1,"PRS_example":0.2}]}'
```

The abbreviated request above is illustrative; the actual keys must exactly
cover every feature returned by `/v1/schema`. Interactive OpenAPI documentation
is available at `/docs` while the server runs.

For stronger filesystem isolation, build the supplied inference-only image from
this directory. Its build copies only the top-level Python files inside the
`dcmfnet/` package. It does not copy the nested `dcmfnet/training/` package,
legacy workflows, audit metadata, or data:

```bash
docker build -f Dockerfile.api -t dcmfnet-api .
docker run --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  -p 8000:8000 \
  --mount type=bind,src=/secure/export/dcmfnet_pos.pt,dst=/model/dcmfnet.pt,readonly \
  dcmfnet-api
```

Mount only the exported `.pt` file, not its `.audit.json` companion or the
directory holding the confidential CSV. The unprivileged API process therefore
has no filesystem route to training or testing datasets.

## Clinical deployment boundary

This is research decision-support software, not a diagnostic system or a
validated clinical risk probability. Before any clinical use, independently
validate discrimination, calibration, subgroup performance, missing-data
behavior, and drift on the intended population. Define clinician-approved
thresholds outside the model only after calibration. Deploy behind appropriate
authentication, authorization, encryption, audit logging, consent, retention,
and applicable medical-device/privacy controls. Do not send direct identifiers
to the endpoint, and do not let an autonomous agent use the score as the sole
basis for care, triage, or treatment.
