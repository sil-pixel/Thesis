"""Check that the updated GLMM/GAMM specifications match the DCMFNet modality design.

The reference is DCMFNet's own feature discovery (dcmfnet.training.data.discover_feature_names) applied to
the real input column names recorded in the ML benchmark log (Models/ML/Results/lazy_predict_training.log).
If that log is missing, the synthetic schema (synthetic_data.expected_columns) is used instead.
The existing coefficient tables of the original GLMM/GAMM (Models/*/Results) are used to check that every
existing main effect is kept and every modality that interacted with SUD15 still does.
Each R script's build_model_spec() is evaluated on those column names, and its formula is parsed with
R's terms(), so the test checks the terms R will actually fit. No data are needed and no model is fitted.

Requires Rscript with the packages used by the scripts (readr, lme4, mgcv, dplyr, ggplot2) and jsonlite.
"""

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Models/DCMFNet/Method"))

from synthetic_data import expected_columns
from dcmfnet.training.data import MODALITY_PATTERNS, discover_feature_names

R_SCRIPTS = {
    "GLMM": ROOT / "Models/GLMM/Method/GLMM_updated_split.R",
    "GAMM": ROOT / "Models/GAMM/Method/GAMM_updated_split.R",
}
ML_LOG = ROOT / "Models/ML/Results/lazy_predict_training.log"
EXISTING_COEFS = {
    "GLMM": {tag: ROOT / f"Models/GLMM/Results/{tag}_GLMM_all_coefs.csv" for tag in ("Pos", "Neg")},
    "GAMM": {tag: ROOT / f"Models/GAMM/Results/GAMM_{tag}_all_parametric_coefs.csv" for tag in ("Pos", "Neg")},
}
EXISTING_GAMM_SMOOTHS = {tag: ROOT / f"Models/GAMM/Results/GAMM_{tag}_smooth_terms.csv" for tag in ("Pos", "Neg")}
ANCHOR = "SUD15"
INDEPENDENT_ONLY = r"batch_.*_x_PC"  # enters only DCMFNet's independent stream, never fused

# Sources a script (main() does not run when sourced) and prints its specification as JSON
R_DRIVER = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressWarnings(suppressPackageStartupMessages(source(args[1])))
column_names <- jsonlite::fromJSON(args[2])
spec <- build_model_spec(column_names)
labels <- function(f) attr(terms(f), "term.labels")
cat(jsonlite::toJSON(list(
  score_blocks = spec$score_blocks,
  prs_vars = spec$PRS_vars,
  interaction_terms = spec$interaction_terms,
  pos_terms = labels(spec$pos_formula),
  neg_terms = labels(spec$neg_formula),
  pos_response = jsonlite::unbox(all.vars(spec$pos_formula)[1]),
  neg_response = jsonlite::unbox(all.vars(spec$neg_formula)[1])
), auto_unbox = FALSE))
"""


def r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    check = subprocess.run(
        ["Rscript", "-e", "q(status = !all(sapply(c('jsonlite','readr','lme4','mgcv','dplyr','ggplot2'), "
                          "requireNamespace, quietly = TRUE)))"],
        capture_output=True,
    )
    return check.returncode == 0


def load_r_spec(script: Path, columns: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        driver = Path(tmp) / "driver.R"
        driver.write_text(R_DRIVER)
        cols = Path(tmp) / "columns.json"
        cols.write_text(json.dumps(columns))
        result = subprocess.run(
            ["Rscript", str(driver), str(script), str(cols)],
            capture_output=True, text=True, check=True,
        )
    return json.loads(result.stdout)


def dataset_columns() -> list[str]:
    """Real column names from the ML benchmark log; synthetic schema as fallback."""
    if ML_LOG.exists():
        with ML_LOG.open() as log:
            for line in log:
                if line.startswith("Input columns:"):
                    inputs = ast.literal_eval(line.split(":", 1)[1].strip())
                    return ["cmpair", *inputs, "SCZ18_Pos_Norm", "SCZ18_Neg_Norm"]
    return expected_columns()


def is_random_effect(term: str) -> bool:
    return "cmpair" in term


def main_effect_variable(term: str) -> str:
    """'s(PRS_x)' -> 'PRS_x'; plain variable names are returned unchanged."""
    return term[2:-1] if term.startswith("s(") and term.endswith(")") else term


@unittest.skipUnless(r_available(), "Rscript with jsonlite, readr, lme4, mgcv, dplyr and ggplot2 is required")
class GlmmGammDcmfnetAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.columns = dataset_columns()
        groups = discover_feature_names(pd.DataFrame(columns=cls.columns))
        cls.dcmfnet = dict(zip(MODALITY_PATTERNS, groups))
        cls.fused = [m for m in MODALITY_PATTERNS if m not in (ANCHOR, INDEPENDENT_ONLY)]
        cls.all_features = [name for group in groups for name in group]
        cls.specs = {model: load_r_spec(script, cls.columns) for model, script in R_SCRIPTS.items()}

    def test_dcmfnet_reference_has_expected_structure(self) -> None:
        self.assertEqual(len(self.fused), 9)
        self.assertEqual(len(self.all_features), 105)

    def test_main_effects_are_exactly_the_dcmfnet_inputs(self) -> None:
        for model, spec in self.specs.items():
            with self.subTest(model=model):
                main_terms = [t for t in spec["pos_terms"] if ":" not in t and not is_random_effect(t)]
                variables = [main_effect_variable(t) for t in main_terms]
                self.assertEqual(len(variables), len(set(variables)), "duplicated main effects")
                self.assertEqual(set(variables), set(self.all_features))

    def test_prs_main_effects_are_smooth_only_in_gamm(self) -> None:
        prs = set(self.dcmfnet["PRS"])
        for model, spec in self.specs.items():
            with self.subTest(model=model):
                main_terms = [t for t in spec["pos_terms"] if ":" not in t and not is_random_effect(t)]
                smooth = {main_effect_variable(t) for t in main_terms if t.startswith("s(")}
                self.assertEqual(smooth, prs if model == "GAMM" else set())

    def test_twin_pair_random_effect_is_included_once(self) -> None:
        expected = {"GLMM": "1 | cmpair", "GAMM": 's(cmpair, bs = "re")'}
        for model, spec in self.specs.items():
            with self.subTest(model=model):
                random_terms = [t for t in spec["pos_terms"] if is_random_effect(t)]
                self.assertEqual(random_terms, [expected[model]])

    def test_modality_scores_use_exactly_the_dcmfnet_items(self) -> None:
        for model, spec in self.specs.items():
            for modality, items in spec["score_blocks"].items():
                with self.subTest(model=model, modality=modality):
                    self.assertIn(modality, self.dcmfnet)
                    self.assertEqual(set(items), set(self.dcmfnet[modality]))
            with self.subTest(model=model, modality="PRS"):
                self.assertEqual(set(spec["prs_vars"]), set(self.dcmfnet["PRS"]))

    def test_interactions_mirror_dcmfnet_fusion(self) -> None:
        for model, spec in self.specs.items():
            with self.subTest(model=model):
                interactions = [t for t in spec["pos_terms"] if ":" in t]
                # R's parser must give the same interaction set as the specification (no silent recycling)
                normalise = lambda terms: {tuple(sorted(t.split(":"))) for t in terms}  # noqa: E731
                self.assertEqual(normalise(interactions), normalise(spec["interaction_terms"]))

                fused_partner_modalities = []
                for term in interactions:
                    parts = term.split(":")
                    self.assertEqual(len(parts), 2, f"not a two-way interaction: {term}")
                    self.assertIn(f"{ANCHOR}_score", parts, f"interaction without the SUD15 anchor: {term}")
                    partner = next(p for p in parts if p != f"{ANCHOR}_score")
                    if partner.endswith("_z") and partner[:-2] in self.dcmfnet["PRS"]:
                        fused_partner_modalities.append("PRS")
                    elif partner in self.dcmfnet["SEX"]:
                        fused_partner_modalities.append("SEX")
                    elif partner.endswith("_score") and partner[:-6] in self.fused:
                        fused_partner_modalities.append(partner[:-6])
                    else:
                        self.fail(f"interaction partner {partner!r} is not a DCMFNet fused modality")

                # Every fused modality interacts with the anchor; batch_x_PC and the anchor itself do not
                self.assertEqual(set(fused_partner_modalities), set(self.fused))
                # One term per modality, except PRS: one per cell-type score
                counts = pd.Series(fused_partner_modalities).value_counts().to_dict()
                for modality in self.fused:
                    expected = len(self.dcmfnet["PRS"]) if modality == "PRS" else 1
                    self.assertEqual(counts[modality], expected, modality)
                self.assertEqual(len(interactions), len(self.fused) - 1 + len(self.dcmfnet["PRS"]))

    def test_no_interaction_with_independent_stream_features(self) -> None:
        batch_pc = set(self.dcmfnet[INDEPENDENT_ONLY])
        for model, spec in self.specs.items():
            with self.subTest(model=model):
                for term in (t for t in spec["pos_terms"] if ":" in t):
                    self.assertFalse(batch_pc & set(term.split(":")), term)

    def test_pos_and_neg_models_share_the_same_terms(self) -> None:
        for model, spec in self.specs.items():
            with self.subTest(model=model):
                self.assertEqual(spec["pos_terms"], spec["neg_terms"])
                self.assertEqual(spec["pos_response"], "SCZ18_Pos_Norm")
                self.assertEqual(spec["neg_response"], "SCZ18_Neg_Norm")

    def modality_of(self, variable: str) -> str:
        return next(m for m, items in self.dcmfnet.items() if variable in items)

    def test_existing_main_effects_are_kept(self) -> None:
        for model, files in EXISTING_COEFS.items():
            new_main = {
                main_effect_variable(t) for t in self.specs[model]["pos_terms"]
                if ":" not in t and not is_random_effect(t)
            }
            for tag, path in files.items():
                with self.subTest(model=model, outcome=tag):
                    if not path.exists():
                        self.skipTest(f"{path} not found")
                    old_main = [v for v in pd.read_csv(path)["Variable"] if ":" not in v]
                    self.assertEqual([v for v in old_main if v not in new_main], [])

    def test_existing_gamm_prs_smooths_are_kept(self) -> None:
        new_smooth = {t for t in self.specs["GAMM"]["pos_terms"] if t.startswith("s(") and not is_random_effect(t)}
        for tag, path in EXISTING_GAMM_SMOOTHS.items():
            with self.subTest(outcome=tag):
                if not path.exists():
                    self.skipTest(f"{path} not found")
                old_smooth = {v for v in pd.read_csv(path)["Variable"] if v.startswith("s(PRS")}
                self.assertTrue(old_smooth)
                self.assertEqual(old_smooth - new_smooth, set())

    def test_existing_sud15_interactions_are_covered_by_modality(self) -> None:
        """Every modality (and every PRS) that interacted with SUD15 before still interacts with SUD15.
        The old item-level pairs were produced by paste0() recycling, so they are matched by modality."""
        for model, files in EXISTING_COEFS.items():
            new_partners = {
                p for t in self.specs[model]["pos_terms"] if ":" in t
                for p in t.split(":") if p != f"{ANCHOR}_score"
            }
            for tag, path in files.items():
                with self.subTest(model=model, outcome=tag):
                    if not path.exists():
                        self.skipTest(f"{path} not found")
                    old = [v for v in pd.read_csv(path)["Variable"] if ":" in v and not v.startswith("factor(")]
                    self.assertTrue(old)
                    for term in old:
                        a, b = term.split(":")
                        partner = b if self.modality_of(a) == ANCHOR else a
                        modality = self.modality_of(partner)
                        expected = f"{partner}_z" if modality == "PRS" else (
                            partner if modality == "SEX" else f"{modality}_score")
                        self.assertIn(expected, new_partners, f"{term} has no SUD15 interaction in the new model")

    def test_glmm_and_gamm_have_the_same_interactions(self) -> None:
        interactions = {
            model: sorted(t for t in spec["pos_terms"] if ":" in t) for model, spec in self.specs.items()
        }
        self.assertEqual(interactions["GLMM"], interactions["GAMM"])


if __name__ == "__main__":
    unittest.main()
