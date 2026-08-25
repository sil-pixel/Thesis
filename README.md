Modeling Schizophrenia Symptom Severity: A Deep Learning and Statistical Approach to Substance Use Interactions in Adolescence

Background: Schizophrenia is a multifactorial disorder shaped by genetic, environmental, and behavioral influences, yet how these factors interact during development remains unclear. Adolescent substance use is a key exposure, but its role as an independent risk factor versus an interaction-driven modifier remains debated.  

Aim: To investigate how adolescent substance use interacts with genetic liability, adverse childhood experiences, and developmental factors to influence schizophrenia symptom severity, and to assess whether a deep learning framework improves modeling of these interactions.

Methods: Using data from the Swedish Twin Registry (CATSS), we applied Generalized Linear Mixed Models (GLMM), Generalized Additive Mixed Models (GAMM), and a Deep Cross-Modal Fusion Network (DCMFNet). The DCMFNet explicitly models interactions between substance use and multiple modalities. Models were evaluated using error-based and rank-based metrics, with interpretability assessed through feature and interaction importance.

Results: Risk was not driven by single factors but by interacting vulnerabilities. Adverse childhood experiences, prior symptoms, and genetic liability formed a consistent baseline, while substance use modified risk selectively. Positive symptoms showed stronger interaction effects, particularly with ACE and cell-type PRS, whereas negative symptoms were dominated by main effects with weaker interactions. DCMFNet provided modest but consistent improvements, especially in capturing rank-based structure.

Conclusion: Schizophrenia symptom severity is best understood through interacting risk factors. Substance use acts as a context-dependent amplifier rather than an independent driver. DCMFNet offers a flexible framework for modeling such interactions, complementing traditional statistical approaches

## Local synthetic DCMFNet workflow

```bash
python generate_synthetic_data.py --n-samples 20000 --seed 42
python train.py
python export_model.py
```

Training outputs both `artifacts/dcmfnet_pos.pt` and `artifacts/dcmfnet_neg.pt`. Export writes both models to `exports/`, each with its corresponding `.metadata.json` file. Copy the exported inference files to the Clinical Risk AI Agent repository and serve the required `.pt` artifacts with its local FastAPI runtime. See `Models/DCMFNet/Method/DEPLOYMENT.md` and `SYNTHETIC_DATA.md`.

## Held-out test metrics

### Positive-symptom model

Artifact: `artifacts/dcmfnet_pos.pt`

- MAE: `0.027113692834973335`
- R²: `0.79314124584198`
- Spearman's ρ: `0.8695505930338211`
- RMSE: `0.03393961518415752`

### Negative-symptom model

Artifact: `artifacts/dcmfnet_neg.pt`

- MAE: `0.05322835221886635`
- R²: `0.3738275170326233`
- Spearman's ρ: `0.6088225454028997`
- RMSE: `0.06683719920868696`
