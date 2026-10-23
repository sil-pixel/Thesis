# Reference Registry

Verification date: 2026-10-14. Metadata was verified via the Crossref REST API (`api.crossref.org/works/{DOI}`) unless stated otherwise. **Content access** records what was actually read: `abstract` = abstract read (OpenAlex/search snippet); `metadata` = bibliographic identity only, content not assessed; `thesis-cited` = cited in the thesis, content not re-read by the agent.
A reference may support a claim in the manuscript **only** if its content access covers that claim. Otherwise use `[CITATION NEEDED]`.

## A. Method / architecture sources

| ID | Reference | DOI / URL | Type | Discovery | Verif. | Content access | Supports | Section |
|---|---|---|---|---|---|---|---|---|
| R-DCMF | Huang Z, Xue M, Liu Y, Xu K, Li J, Yu C. DCMFNet: Deep cross-modal fusion network for different modalities with iterative gated fusion. In: Proc. Graphics Interface 2024. p. 1–12. | 10.1145/3670947.3670956 | Conf. proceedings (peer-reviewed) | Thesis [16] | ✅ Crossref | metadata | Origin of IGF/ASGate design; default 5 layers `[VERIFY: default hyperparameters stated in paper]` | Intro, Methods |
| R-SE | Hu J, Shen L, Sun G. Squeeze-and-excitation networks. In: Proc. IEEE/CVF CVPR 2018. p. 7132–41. | 10.1109/CVPR.2018.00745 | Conf. | Thesis [20] (arXiv) → upgraded | ✅ | metadata (known content: channel squeeze by global pooling) | Inspiration for gating; contrast with implemented operation | Methods |
| R-TABNET | Arik SÖ, Pfister T. TabNet: attentive interpretable tabular learning. Proc AAAI Conf Artif Intell 2021;35(8):6679–87. | 10.1609/aaai.v35i8.16826 | Conf./journal | New | ✅ | metadata | Prior attention-based tabular DL (baseline candidate) | Intro, Methods |
| R-GRIN | Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? In: Advances in Neural Information Processing Systems 35 (NeurIPS 2022 Datasets & Benchmarks). | arXiv:2207.08815; https://neurips.cc/virtual/2022/poster/55627 | Conf. | New (web search) | ✅ (NeurIPS listing + arXiv) | abstract | Tree ensembles outperform DL on medium-sized (~10k) tabular data | Intro, Discussion |
| R-SHWARTZ | Shwartz-Ziv R, Armon A. Tabular data: deep learning is not all you need. Inf Fusion 2022;81:84–90. | 10.1016/j.inffus.2021.11.011 | Journal | New | ✅ | metadata | DL vs GBDT on tabular data `[VERIFY content before citing specific finding]` | Intro, Discussion |
| R-LGBM | Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, et al. LightGBM: a highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems 30 (NIPS 2017). p. 3146–54. | https://papers.nips.cc/paper/6907 `[VERIFY URL/pages]` | Conf. | Replaces thesis [19] | ⚠️ (no DOI; Crossref miss) | metadata | LightGBM used for feature screening | Methods |
| R-SHAP | Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems 30 (NIPS 2017). p. 4765–74. | `[VERIFY URL/pages]` | Conf. | New | ⚠️ (no DOI) | metadata | SHAP used in screening | Methods |
| R-OPTUNA | Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. In: Proc. 25th ACM SIGKDD. 2019. p. 2623–31. | 10.1145/3292500.3330701 | Conf. | New | ✅ | metadata | Hyperparameter search tool (TPE sampler default `[VERIFY: sampler actually used = default TPESampler]`) | Methods |
| R-LAZY | Pandala SR. LazyPredict. Version 0.3.0 [software]. 2026. | https://github.com/shankarpandala/lazypredict | Software | Thesis [21] | ⚠️ `[VERIFY version used and release date]` | — | Baseline benchmarking tool | Methods |
| R-PRSCS | Ge T, Chen C-Y, Ni Y, Feng Y-CA, Smoller JW. Polygenic prediction via Bayesian regression and continuous shrinkage priors. Nat Commun 2019;10:1776. | 10.1038/s41467-019-09718-5 | Journal | Thesis [17] | ✅ | metadata | PRS-CS method | Methods |
| R-JAIN | Jain S, Wallace BC. Attention is not explanation. In: Proc. NAACL-HLT 2019. p. 3543–56. | 10.18653/v1/N19-1357 | Conf. | New | ✅ | metadata (well-known content) | Gate/attention weights are not faithful explanations | Methods, Discussion |

## B. Clinical prediction methodology / reporting

| ID | Reference | DOI | Type | Verif. | Content | Supports | Section |
|---|---|---|---|---|---|---|---|
| R-TRIPODAI | Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. | 10.1136/bmj-2023-078378 | Journal | ✅ | metadata | Reporting standard | Methods, Supplement |
| R-CHRIST | Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol 2019;110:12–22. | 10.1016/j.jclinepi.2019.02.004 | Journal (SR) | ✅ | abstract | 71 studies; ML not better than LR in low-bias comparisons; calibration rarely assessed | Intro, Discussion |
| R-RILEY | Riley RD, Ensor J, Snell KIE, Harrell FE, Martin GP, Reitsma JB, et al. Calculating the sample size required for developing a clinical prediction model. BMJ 2020;368:m441. | 10.1136/bmj.m441 | Journal | ✅ | metadata | Sample-size reasoning for parameters vs n | Discussion |
| R-VANCAL | Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med 2019;17:230. | 10.1186/s12916-019-1466-7 | Journal | ✅ | metadata | Need for calibration assessment | Methods |
| R-KAPOOR | Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 2023;4:100804. | 10.1016/j.patter.2023.100804 | Journal | ✅ | metadata | Leakage taxonomy (feature selection / model selection on test) | Methods, Limitations |
| R-NADEAU | Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn 2003;52:239–81. | 10.1023/A:1024068626366 | Journal | ✅ | metadata | Corrected resampled t-test for repeated random splits | Methods (statistics) |

## C. Psychiatric epidemiology / cohort / genetics

| ID | Reference | DOI | Type | Verif. | Content | Supports | Section |
|---|---|---|---|---|---|---|---|
| R-CATSS | Anckarsäter H, Lundström S, Kollberg L, Kerekes N, Palm C, Carlström E, et al. The Child and Adolescent Twin Study in Sweden (CATSS). Twin Res Hum Genet 2011;14:495–508. | 10.1375/twin.14.6.495 | Journal | ✅ | metadata | Cohort description | Methods §2.1 |
| R-TATE | Tate AE, McCabe RC, Larsson H, Lundström S, Lichtenstein P, Kuja-Halkola R. Predicting mental health problems in adolescence using machine learning techniques. PLoS One 2020;15:e0230389. | 10.1371/journal.pone.0230389 | Journal | ✅ | thesis-cited (thesis: "comparable performance to traditional regression") `[VERIFY: read abstract/full text]` | Closest prior ML work on CATSS | Intro, Discussion |
| R-SULLIVAN | Sullivan PF, Kendler KS, Neale MC. Schizophrenia as a complex trait: evidence from a meta-analysis of twin studies. Arch Gen Psychiatry 2003;60:1187–92. | 10.1001/archpsyc.60.12.1187 | Journal | ✅ (author order corrected vs thesis) | thesis-cited | Heritability ~81% `[VERIFY exact estimate before citing]` | Intro |
| R-PGC3 | Trubetskoy V, Pardiñas AF, Qi T, Panagiotaropoulou G, Awasthi S, Bigdeli TB, et al. Mapping genomic loci implicates genes and synaptic biology in schizophrenia. Nature 2022;604:502–8. | 10.1038/s41586-022-04434-5 | Journal | ✅ | metadata | Schizophrenia GWAS (PRS training source?) `[AUTHOR INPUT: which GWAS summary statistics fed PRS-CS]` | Methods §2.3 |
| R-YAO2025 | Yao S, Harder A, Darki F, Chang Y, Li A, Nikouei K, et al. Connecting genomic results for psychiatric disorders to human brain cell types and regions reveals convergence with functional connectivity. Nat Commun 2025;16:395. | 10.1038/s41467-024-55611-1 | Journal | ✅ | thesis-cited | Cell-type enrichment underlying cell-type PRS | Intro, Methods |
| R-SILETTI | Siletti K, Hodge R, Mossi Albiach A, Lee K, Ding S, Hu L, et al. Transcriptomic diversity of cell types across the adult human brain. Science 2023;382:eadd7046. | 10.1126/science.add7046 | Journal | ✅ | thesis-cited | Cell-type atlas for gene sets | Methods §2.3 |
| R-WANG2025 | Wang H. The association between mechanism-informed polygenic risk score and schizophrenia and treatment resistance [master's thesis]. Stockholm: Karolinska Institutet; 2025. | — | Thesis (grey literature) | ⚠️ not verifiable online | — | PRS pipeline origin. AIIM discourages unpublished sources: cite only if it is accessible, otherwise describe the pipeline fully in a Supplement. | Methods |
| R-DIFORTI | Di Forti M, Quattrone D, Freeman TP, Tripoli G, Gayer-Anderson C, Quigley H, et al. The contribution of cannabis use to variation in the incidence of psychotic disorder across Europe (EU-GEI). Lancet Psychiatry 2019;6:427–36. | 10.1016/S2215-0366(19)30048-3 | Journal | ✅ | metadata (well-known) | Cannabis–psychosis association | Intro |
| R-HALL | Hall W, Degenhardt L. Cannabis use and the risk of developing a psychotic disorder. World Psychiatry 2008;7:68–71. | 10.1002/j.2051-5545.2008.tb00158.x | Journal | ✅ | thesis-cited | Cannabis–psychosis association | Intro |
| R-VARESE | Varese F, Smeets F, Drukker M, Lieverse R, Lataster T, Viechtbauer W, et al. Childhood adversities increase the risk of psychosis: a meta-analysis of patient-control, prospective- and cross-sectional cohort studies. Schizophr Bull 2012;38:661–71. | 10.1093/schbul/sbs050 | Journal (MA) | ✅ | metadata (well-known) | Adversity–psychosis association | Intro |
| R-ROBINSON | Robinson N, Bergen SE. Environmental risk factors for schizophrenia and bipolar disorder and their relationship to genetic risk: current knowledge and future directions. Front Genet 2021;12:686666. | 10.3389/fgene.2021.686666 | Journal (review) | ✅ | thesis-cited | G×E framing | Intro |
| R-AGERBO | Agerbo E, Sullivan PF, Vilhjálmsson BJ, Pedersen CB, Mors O, Børglum AD, et al. Polygenic risk score, parental socioeconomic status, family history of psychiatric disorders, and the risk for schizophrenia: a Danish population-based study and meta-analysis. JAMA Psychiatry 2015;72:635–41. | 10.1001/jamapsychiatry.2015.0346 | Journal | ✅ | thesis-cited | PRS + SES | Intro |
| R-SMFQ | Angold A, Costello EJ. Short Mood and Feelings Questionnaire [database record]. APA PsycTests; 1987. | 10.1037/t15197-000 | Instrument record | ✅ | metadata | **Only if** the author confirms the depressive outcome items come from SMFQ (D-03) | Methods §2.2 |

## D. AIIM papers (see `aiim_paper_analysis.md`)

| ID | Reference | DOI | Verif. | Content |
|---|---|---|---|---|
| R-BORSOS | Borsos B, Allaart CG, van Halteren A. Predicting stroke outcome: a case for multimodal deep learning methods with tabular and CT perfusion data. Artif Intell Med 2024;147:102719. | 10.1016/j.artmed.2023.102719 | ✅ Crossref/OpenAlex | abstract |
| R-SHEIKH | Sheikhalishahi S, Bhattacharyya A, Celi LA, Osmani V. An interpretable deep learning model for time-series electronic health records: case study of delirium prediction in critical care. Artif Intell Med 2023;144:102659. | 10.1016/j.artmed.2023.102659 | ✅ | abstract |
| R-EIJPE | Eijpe A, Lakbir S, Cesur ME, Oliveira SP, Chatzimparmpas A, Abeln S, et al. Bridging the gap between performance and interpretability: an explainable disentangled multimodal framework for cancer survival prediction. Artif Intell Med 2026;181:103504. | 10.1016/j.artmed.2026.103504 | ✅ | abstract |
| R-FOTEIN | Foteinopoulou NM, Patras I. Machine learning approaches for fine-grained symptom estimation in schizophrenia: a comprehensive review. Artif Intell Med 2025;165:103129. | 10.1016/j.artmed.2025.103129 | ✅ | abstract |
| R-HANCOX | Hancox Z, Pang A, Conaghan PG, Kingsbury SR, Clegg A, Relton SD. A systematic review of networks for prognostic prediction of health outcomes and diagnostic prediction of health conditions within electronic health records. Artif Intell Med 2024;158:102999. | 10.1016/j.artmed.2024.102999 | ✅ | abstract |
| R-ANDREEVA | Andreeva O, Ding WY, Leveille SG, Cai Y, Chen P. Fall risk assessment through a synergistic multi-source DNN learning model. Artif Intell Med 2022;127:102280. | 10.1016/j.artmed.2022.102280 | ✅ | metadata only |
| R-WANGZ | Wang Z, Poon J, Wang S, Sun S, Poon SK. A novel method for clinical risk prediction with low-quality data. Artif Intell Med 2021;114:102052. | 10.1016/j.artmed.2021.102052 | ✅ | metadata only |
| R-BASNET | Basnet S, Parasteh S, Manashty A, Sasyniuk B. RIMD: a novel method for clinical prediction. Artif Intell Med 2023;140:102526. | 10.1016/j.artmed.2023.102526 | ✅ | metadata only |
| R-BERTL | Bertl M, Bignoumba N, Ross P, Ben Yahia S, Draheim D. Evaluation of deep learning-based depression detection using medical claims data. Artif Intell Med 2024;147:102745. | 10.1016/j.artmed.2023.102745 | ✅ | metadata only |
| R-VALENTE | Valente F, Henriques J, Paredes S, Rocha T, Carvalho P, Morais J. A new approach for interpretability and reliability in clinical risk prediction: acute coronary syndrome scenario. Artif Intell Med 2021;117:102113. | 10.1016/j.artmed.2021.102113 | ✅ | metadata only |
| R-BARROS | Barros C, Silva CA, Pinheiro AP. Advanced EEG-based learning approaches to predict schizophrenia: promises and pitfalls. Artif Intell Med 2021;114:102039. | 10.1016/j.artmed.2021.102039 | ✅ | metadata only |
| R-ENGEL | Engelhard M, Wojdyla DM, Wang H, Pencina MJ, Henao R. Exploring trade-offs in equitable stroke risk prediction with parity-constrained and race-free models. Artif Intell Med 2025;164:103130. | 10.1016/j.artmed.2025.103130 | ✅ | metadata only |

## E. Thesis references not carried forward (pending review)
- Thesis [1] Robison et al. 2020 (Biol Psychiatry); [2] Zhan et al. 2023 (Front Genet); [4] Luvsannyam et al. 2022 (Cureus); [8] Martins et al. 2014; [9] Shannon et al. 2020 (conference abstract); [10] Jutla et al. 2022; [11] Dalsgaard et al. 2014; [13] Carter et al. 2022. These are background-only and secondary for an AIIM paper. [4] (Cureus review) and [9] (meeting abstract) are weak sources and should be replaced if the claims are needed.

## F. Needed but not yet found / verified
- `[CITATION NEEDED: imbalanced regression reweighting — e.g., Yang Y et al., Delving into deep imbalanced regression, ICML 2021]`
- `[CITATION NEEDED: FT-Transformer / Gorishniy et al., Revisiting deep learning models for tabular data, NeurIPS 2021]` (Crossref search failed; verify via proceedings)
- `[CITATION NEEDED: psychotic-like experiences in adolescence and later psychopathology, CATSS-based (e.g., Kelleher/Cederlöf et al. 2017 Schizophr Res 181:13–16, doi 10.1016/j.schres.2016.08.029 — found, content unverified)]`
- `[CITATION NEEDED: prior ML/DL prediction using PRS + environment for psychosis/depression outcomes (direct competitors)]`
- `[CITATION NEEDED: twin/family dependence and grouped cross-validation leakage]`
