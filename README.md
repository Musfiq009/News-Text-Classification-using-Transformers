<div align="center">

# 📰 News Text Classification Across Two Datasets
### BERT, DistilBERT and Ensemble Models

**A comparative NLP study of BERT, DistilBERT and a BERT–DistilBERT ensemble on two news datasets.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-BERT%20%7C%20DistilBERT-FFD21E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)

*Natural Language Processing — Project · Summer 2025–2026*
*American International University-Bangladesh (AIUB) · Faculty of Science & Technology · Department of Computer Science*

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Datasets](#-datasets)
3. [Pipeline](#-pipeline)
4. [Experiments](#-experiments)
5. [Results](#-results)
6. [Key Findings](#-key-findings)
7. [Tech Stack](#-tech-stack)
8. [Getting Started](#-getting-started)
9. [Limitations](#-limitations)
10. [Future Work](#-future-work)

---

## 📖 Overview

This project studies **document type detection through news text classification** on two datasets with different structure and size. Because the datasets differ in category structure and data characteristics, models may not behave the same way on both — which makes them a good test bed for comparing classification approaches.

Transformer models (**BERT**, **DistilBERT**) were fine-tuned with different hyper-parameter combinations, and a **soft-voting ensemble** of the two was evaluated.

The experiments aim not only to compare models, but also to observe how **parameter selection** and **dataset differences** affect performance.

---

## 🗂️ Datasets

| | Dataset 1 | Dataset 2 |
|---|-----------|-----------|
| **Source** | BBC News articles | News Category dataset |
| **Size** | 2,225 articles | 252 articles |
| **Classes** | 5 | 8 |
| **Categories** | sport (511), business (510), politics (417), tech (401), entertainment (386) | Entertainment (39), Food (36), Economy (35), Sports (32), International relations (32), Health (27), Artificial Intelligence (27), Politics (24) |
| **Columns** | `category`, `text` | `News`, `Category` |

Dataset 2 is far smaller and has more classes, which makes it much more sensitive to training settings.

---

## 🔄 Pipeline

```
Raw text ─► Case folding ─► Punctuation removal ─► Tokenization (NLTK)
         ─► Stopword removal ─► Lemmatization (spaCy en_core_web_sm) ─► clean_text
                                      │
                                      ▼
                  Transformers (Hugging Face Trainer)
              BERT · DistilBERT ─► Soft-voting ensemble
                                      │
                                      ▼
                  Accuracy · Precision · Recall · F1 (weighted)
```

**Preprocessing** (both datasets): lower-casing, punctuation removal, tokenization, stopword removal and lemmatization, saved as a `clean_text` column.

**Data cleaning before transformer training:** required-column check, drop missing/empty rows, strip whitespace, lower-case labels, remove duplicate texts, label encoding (`label2id` / `id2label`).

---

## 🧪 Experiments

### 1. Selecting the best BERT configuration
A grid over **learning rate × batch size × weight decay** was run on both datasets.

| Fixed settings | Searched values |
|----------------|-----------------|
| Max sequence length 128 | Learning rate: `2e-5`, `3e-5` |
| Dropout 0.1 | Batch size: `16`, `32` |
| Warm-up ratio 0.1 | Weight decay: `0.01`, `0.1` |
| 5 epochs, AdamW | Early stopping (patience 2), metric for best model: F1 |

### 2. Selecting the best DistilBERT configuration
Same grid and fixed settings as BERT.

### 3. Ensemble (soft voting)
Class probabilities from BERT and DistilBERT (softmax of logits) are **averaged with equal weight**; the class with the highest averaged probability is the final prediction.

### Selected configurations

| Model | Learning rate | Batch size | Weight decay |
|-------|:-------------:|:----------:|:------------:|
| BERT | 0.00003 | 16 | 0.01 |
| DistilBERT | 0.00003 | 16 | 0.1 |
| Ensemble | 0.00003 | 16 | 0.01 |

---

## 📊 Results

### BERT hyper-parameter search

| LR | Batch | WD | D1 Acc | D1 F1 | D2 Acc | D2 F1 |
|----|:-----:|:--:|:------:|:-----:|:------:|:-----:|
| 0.00002 | 16 | 0.01 | 97.64% | 97.65% | 64.71% | 61.67% |
| **0.00003** | **16** | **0.01** | **98.58%** | **98.58%** | **66.67%** | **62.92%** |
| 0.00002 | 32 | 0.01 | 98.58% | 98.59% | 31.37% | 24.53% |
| 0.00003 | 32 | 0.01 | 99.53% | 99.53% | 41.18% | 30.82% |
| 0.00002 | 16 | 0.1 | 98.11% | 98.11% | 64.71% | 61.67% |
| 0.00003 | 16 | 0.1 | 98.58% | 98.58% | 66.67% | 62.92% |
| 0.00002 | 32 | 0.1 | 98.58% | 98.59% | 31.37% | 24.53% |
| 0.00003 | 32 | 0.1 | 98.58% | 98.59% | 41.18% | 30.82% |

### DistilBERT hyper-parameter search

| LR | Batch | WD | D1 Acc | D1 F1 | D2 Acc | D2 F1 |
|----|:-----:|:--:|:------:|:-----:|:------:|:-----:|
| 0.00002 | 16 | 0.01 | 98.58% | 98.59% | 65.38% | 59.46% |
| 0.00003 | 16 | 0.01 | 97.64% | 97.64% | 80.77% | 79.54% |
| 0.00002 | 32 | 0.01 | 98.58% | 98.58% | 46.15% | 30.83% |
| 0.00003 | 32 | 0.01 | 98.58% | 98.58% | 57.69% | 51.03% |
| 0.00002 | 16 | 0.1 | 99.06% | 99.05% | 65.38% | 59.46% |
| **0.00003** | **16** | **0.1** | **98.58%** | **98.58%** | **80.77%** | **79.54%** |
| 0.00002 | 32 | 0.1 | 98.58% | 98.58% | 46.15% | 30.83% |
| 0.00003 | 32 | 0.1 | 98.58% | 98.58% | 57.69% | 51.21% |

### Overall comparison

| Model | D1 Acc | D1 Prec | D1 Rec | D1 F1 | D2 Acc | D2 Prec | D2 Rec | D2 F1 |
|-------|:------:|:-------:|:------:|:-----:|:------:|:-------:|:------:|:-----:|
| BERT | 98.58% | 98.58% | 98.58% | 98.58% | 66.67% | 69.39% | 66.67% | 62.92% |
| DistilBERT | 98.58% | 98.61% | 98.58% | 98.58% | 80.77% | 80.51% | 80.77% | 79.54% |
| Ensemble | 98.58% | 98.58% | 98.58% | 98.58% | 76.92% | 68.02% | 76.92% | 71.45% |

> DistilBERT gives the best result on Dataset 2 (80.77% accuracy, 79.54% F1).

---

## 💡 Key Findings

- **Dataset 1 is solved well by all transformers** — BERT, DistilBERT and the ensemble all reach **98.58%** accuracy.
- **Dataset 2 is far more sensitive** — transformers react strongly to batch size and learning rate (e.g. BERT drops to 31–41% with batch size 32).
- **The best configuration for one dataset is not the best for both** — BERT with batch size 32 reached 99.53% on Dataset 1 but only 41.18% on Dataset 2, so batch size 16 was selected.
- **DistilBERT is the most balanced model** — a smaller, lighter model gave the best result on Dataset 2.
- **Ensembling is not automatically better** — the ensemble improved on BERT for Dataset 2 (76.92% vs 66.67%) but did not beat DistilBERT (80.77%).

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Preprocessing | pandas, NumPy, NLTK (`punkt`, `stopwords`), spaCy (`en_core_web_sm`) |
| Deep learning | PyTorch, Hugging Face Transformers (`Trainer`, `EarlyStoppingCallback`) |
| Evaluation | scikit-learn metrics (accuracy, weighted precision / recall / F1) |
| Environment | Kaggle GPU notebooks (BERT / DistilBERT / ensemble) |

---

## 🚀 Getting Started

> The project was developed in Jupyter notebooks. File names below follow the notebooks in the report — adjust them to match your repository.

**1. Clone and install**

```bash
git clone https://github.com/Musfiq009/News-Text-Classification-using-Transformers.git
cd News-Text-Classification-using-Transformers
pip install pandas numpy nltk spacy scikit-learn torch transformers
python -m spacy download en_core_web_sm
```

**2. Add the data** — place the BBC News CSV (`bbc-text.csv`) and the News Category CSV (`News-Categoires.csv`) in your working or Drive folder.

**3. Run the notebooks in order**

| Step | Notebook | Purpose |
|:----:|----------|---------|
| 1 | `NLP_PROJECT_PREPROCESSING.ipynb` | Clean text and export `bbc-text-preprocessed.csv` (same steps for Dataset 2) |
| 2 | `bert-1-d1-dataset1.ipynb` | BERT grid search on Dataset 1 (and the matching notebook for Dataset 2) |
| 3 | `distilbert-2-d2-dataset2.ipynb` | DistilBERT grid search (and the matching notebook for Dataset 1) |
| 4 | `ensemble-dataset1-*.ipynb` | Train both models and soft-vote their probabilities |

A GPU is recommended for the transformer notebooks (mixed precision `fp16` is enabled automatically when CUDA is available).

---

## ⚠️ Limitations

- Experiments cover **news text only**, so results may not transfer to other document types or domains.
- **One configuration was selected for both datasets**; it is a good compromise but not necessarily the best for each dataset separately.
- **Dataset 2 is small (252 rows)** and its results are more sensitive to parameter changes, making model selection less stable.
- The ensemble gives **equal weight** to BERT and DistilBERT; another combination method could change the outcome.

---

## 🔭 Future Work

- Test the same approach on other kinds of text and document classification tasks.
- Tune hyper-parameters per dataset, and use k-fold cross-validation for the small Dataset 2.
- Try weighted or learned ensembling instead of equal-weight soft voting.
- Expand Dataset 2 or apply data augmentation to improve stability.
- Evaluate additional transformer models (e.g. RoBERTa, ALBERT).

---


