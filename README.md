# PanVision

Explainable AI-assisted pancreatic CT analysis. NUS ISS Graduate Certificate, Pattern Recognition Systems practice project (Group 3).

PanVision is an **academic research prototype**. It is not a clinical tool and gives no diagnosis or treatment recommendation.

## What it does

Upload a portal-venous abdominal CT (`.nii`, `.nii.gz` or `.mha`) and PanVision runs this pipeline:

```text
CT upload ─▶ preprocess ─▶ pancreas segmentation ─▶ fixed 144 mm crops ─▶ PDAC classifier
 (validated)  (1.5×1.5×3 mm,   (nnU-Net v2, 2D)        around the predicted     (DenseNet-121,
              -100..300 HU)                             pancreas, 224×224        patient-level score)
                                                                  │
              lesion segmentation (attention U-Net)               ├─▶ Grad-CAM on the most suspicious slice
              shown as a research overlay                         └─▶ Top-3 similar training cases (cosine KNN
                                                                      on 512-number embeddings, stored in Neon)
                                  all results ─▶ Streamlit dashboard + PDF case report
```

1. **Preprocessing** (`src/preprocessing/`): PANORAMA portal-venous CT, LPS orientation, 1.5 × 1.5 × 3 mm spacing, HU window −100 to 300, one shared **patient-level** split (`data/splits.csv`).
2. **Segmentation** (`src/segmentation/`): pancreas with nnU-Net v2 (2D); lesion with an attention U-Net on the pancreas region. The lesion model is weak and also marks regions in scans without PDAC, so it is shown as a research overlay only.
3. **Classification** (`src/classification/`): DenseNet-121 on 2D crops (fixed 144 mm window) cut around the **predicted** pancreas. The patient score is the mean of the slice probabilities; the cut-off was chosen on validation patients and then frozen. ResNet-50 and a small CNN were trained as comparisons.
4. **Explainability** (`src/explainability/`, `src/knn/`): Grad-CAM shows where the classifier looked (a coarse 7×7 map, not a tumour mask). The Top-3 similar cases are the nearest **training** patients by cosine similarity of the 512-number embeddings; a case never retrieves itself.
5. **Application** (`app/`): FastAPI backend, Streamlit dashboard, PostgreSQL (Neon) for accounts, upload history and the embedding bank, PDF case report.

## Results at a glance

All figures use the shared patient-level split. See the final report for methods, confidence intervals and caveats.

| Part | Result | Notes |
|---|---|---|
| Pancreas segmentation (nnU-Net v2, 2D) | Validation Dice 0.819 (80 cases; U-Net baselines 0.764 and 0.783) | Reference masks are mostly automatic, so this is agreement with an automatic reference |
| PDAC classifier, ResNet-50 | Test AUC 0.882 (95% CI 0.840–0.920); sensitivity 0.84, specificity 0.81 | Crops chosen with reference lesion masks, so absolute values are optimistic |
| PDAC classifier, DenseNet-121 | Validation AUC 0.933 on the same crops; **0.91 on predicted-pancreas crops** (sensitivity 84%, specificity 85%); **test AUC 0.891** (sensitivity 0.84, specificity 0.83) | Selected on validation, then scored once on the test set; deployed in the app |
| Similar-case retrieval (DenseNet) | Precision@3 0.86 on validation queries (chance 0.58) | Offline evaluation, `src/explainability/` |
| Lesion segmentation | Weak | High false-positive rate; not a detector |

## Limitations

- The classifier was developed on crops that used reference lesion masks to pick slices, so absolute numbers are optimistic. Deployment uses predicted-pancreas crops, which do not use any lesion information.
- The test set is small, so the confidence intervals are wide. DenseNet was selected on validation and scored once on the test set (AUC 0.891).
- The lesion model is weak. Grad-CAM is coarse and shows classifier attention, not lesion extent.
- No check for scanner or site effects was done. One dataset only (PANORAMA).

## Repository layout

| Folder | Purpose |
|---|---|
| `data/` | `splits.csv` and notes. Scans, masks and crops are **not** in Git. |
| `src/preprocessing/` | CT loading, windowing, resampling, patient-level split |
| `src/segmentation/` | Pancreas and lesion models, evaluation, measurements |
| `src/classification/` | DenseNet-121, ResNet-50 and CNN classifiers, crop making, evaluation, embedding export |
| `src/explainability/` | Grad-CAM, `explain_study`, offline retrieval evaluation, localisation check |
| `src/knn/` | Similar-case search and loading embeddings into Neon |
| `src/report/` | Template-based summary text |
| `app/` | FastAPI backend, Streamlit dashboard, tests (see `app/README.md`) |
| `deploy/`, `Dockerfile`, `docker-compose.yml`, `.github/` | Deployment to a Google Cloud VM (see `deploy/README.md`) |
| `notebooks/` | Colab notebooks for training and evaluation |
| `docs/` | Report, slides and other supporting documents |

## Run it

**Live demo:** https://34-177-82-16.sslip.io/. It runs on a small Google Cloud VM, so segmentation and classification take a few minutes per scan.
**Initial Login: **
- Email : Admin@gmail.com
- Password: Helloworld1234

**Locally** (Python 3.12, Windows):

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r app/requirements.txt
```

Then follow `app/README.md`: configure the database (`.env`, never committed), start FastAPI and Streamlit, and, for the full analysis, install `app/inference-requirements.txt` and place the model files. Without the model files the app still starts and shows model output as unavailable.

**Tests:**

```text
python -m unittest discover -s app/tests
python -m unittest discover -s src/classification/tests
python -m unittest discover -s src/segmentation/tests -t .
```

**Experiments:** the Colab notebooks in `notebooks/` reproduce training and evaluation. The explainability checks are `notebooks/explainability_densenet.ipynb` and `explainability_resnet.ipynb`; the commands are in `src/explainability/README.md`.

## Model files and data

These are too large for GitHub and are kept out of Git (`.gitignore`):

| What | Where it lives | Expected location in a local run |
|---|---|---|
| Pancreas model (nnU-Net) | Team Drive, deployment VM | `data/private/models/P3_nnunet_2d_full/…` or `PANVISION_PANCREAS_MODEL_DIR` |
| Lesion model | Team Drive, deployment VM | `data/private/models/L2_attention_unet_roi_full/best.pt` or `PANVISION_LESION_CHECKPOINT` |
| Classifier (DenseNet-121, `best.pt` + config) | Team Drive, deployment VM | `data/private/models/classification/` or `PANVISION_CLASSIFIER_DIR` |
| CT data, masks, crops | Team OneDrive (`PanVisionData`) | see `data/README.md` |

The data come from the public PANORAMA challenge dataset; see the report for the reference and how our subset and split were built.

## Team

| Member | Main areas |
|---|---|
| Miti | PANORAMA data preparation, validation, preprocessing, patient-level splitting and QA; custom CNN and DenseNet-121 training, evaluation and embedding generation; PostgreSQL/Neon schema setup and pgvector-backed KNN similar-case retrieval; original Streamlit dashboard, FastAPI backend, authentication, role-based administration and audit history; end-to-end inference orchestration from CT upload and preprocessing through pancreas segmentation, predicted-pancreas cropping, DenseNet classification, patient-level aggregation, embedding extraction, top-3 retrieval, Grad-CAM and reprot generation|
| Akshata | Pancreas and lesion segmentation cascade (U-Net, nnU-Net P3), lesion optimisation (threshold and size tuning, augmentation experiments, L2/L5 ensemble), integrating segmentation, lesion inference and the DenseNet-121 classifier into the upload workflow, API, dashboard and report, review notes, dashboard UI improvements, Docker deployment to GCP Compute Engine via GitHub Actions CI/CD, documentation |
| Hrihaan | Repository setup, ResNet-50 classifier, explainability (Grad-CAM, similar-case evaluation), documentation |

Working rules: Never commit CT data, model weights, `.env` files or connection strings.

## Deliverables

| Deliverable | Where |
|---|---|
| Runnable system | This repository and the live demo above |
| Final report | `<add link or file in docs/>` |
| Presentation video | `https://drive.google.com/file/d/1-nngIBeUh6E44mQhegeCWoJcx_bzIxkW/view?usp=drive_link` |
| Slides | `<add link or file in docs/>` |
