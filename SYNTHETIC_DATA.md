# Synthetic DCMFNet data

`generate_synthetic_data.py` creates a CSV with the exact ordered schema consumed by the local DCMFNet loader: 6 SUD15, 16 PRS, 23 SCZ15, 19 ADHD9, 17 ASD9, 7 ACE15, 4 ACE18, 4 SUD18, 4 SES, 1 SEX, and 4 batch/PC features, plus `cmpair` and both normalized targets. Missing values are deliberate and are imputed by the existing training pipeline.

```bash
python generate_synthetic_data.py --n-samples 20000 --seed 42
```

Default output: `data/synthetic_dcmfnet.csv`.

This dataset is fully synthetic and intended only for development, testing, and demonstration. It does not contain STR participant records and must not be interpreted as clinically representative data.
