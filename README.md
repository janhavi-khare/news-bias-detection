# News Bias Detection

An academic AI/ML project and full-stack system for detecting bias in news text using a fine-tuned **DeBERTa-v3-base binary text classifier** with **SHAP word-level explainability** and an interactive React web interface.

---

## 1. Project Overview & Scope

News media framing, word choice, and tonal sentiment frequently introduce subjective bias into reporting. This project provides an end-to-end machine learning system to classify whether an excerpt of news text exhibits **Biased** or **Non-biased** language.

### Model Description
> **DeBERTa-v3-base is a pretrained transformer encoder fine-tuned on the BABE dataset for sentence-level news media bias detection. Given an English news sentence, it predicts whether the sentence contains biased language or is non-biased. The model is evaluated using accuracy, macro-F1, class-wise F1, precision, recall, and a confusion matrix.**

The model detects potentially biased language. It does not establish whether a news claim is factually true, determine the author's intentions, or prove that an entire article or news outlet is politically biased.

### Current Production Model Specifications
- **Model Architecture**: DeBERTa-v3-base (`microsoft/deberta-v3-base`, `DebertaV2ForSequenceClassification`).
- **Fine-Tuning Dataset**: BABE (sentence-level news media bias detection).
- **Training Approach**: Supervised binary classification (`0 = Non-biased`, `1 = Biased`).
- **Dataset Splits (Article-Disjoint)**:
  - **Training**: 2,579 sentences
  - **Validation**: 550 sentences
  - **Test**: 508 sentences
- **Granularity & Aggregation**: The model classifies text at sentence-level granularity (matching the BABE training set). For multi-sentence articles, it evaluates each sentence independently and derives an article-level bias density summary using a 25% threshold rule.
- **Active Checkpoint**: Saved locally under `model_deberta/`.
- **Interpretability**: Word-level SHAP (`shap.maskers.Text`) token attributions targeting class index 1 (Biased).
- **Important Boundaries**: The model detects *lexical and framing bias* (emotional pejoratives and loaded language). It does **not** classify political leaning (Left/Center/Right), estimate publisher ideology, reliably catch subtle unquoted editorial bias, or verify factual truth.

---

## 2. Authoritative Benchmark Evaluation Results

The production DeBERTa-v3-base model was evaluated on the held-out article-disjoint test split (**508 sentences**) and validation split (**550 sentences**):

| Metric | Split | Value |
| :--- | :--- | :--- |
| **Test Accuracy** | Test (508 sentences) | **82.87%** (0.8287) |
| **Test Macro-F1** | Test (508 sentences) | **82.87%** (0.8287) |
| **Validation Macro-F1** | Validation (550 sentences) | **83.09%** (0.8309) |
| **Non-biased Class F1** | Test (252 sentences) | **82.91%** (0.8291) |
| **Biased Class F1** | Test (256 sentences) | **82.84%** (0.8284) |
| **Biased-Class Precision** | Test (256 sentences) | **83.67%** (0.8367) |
| **Biased-Class Recall** | Test (256 sentences) | **82.03%** (0.8203) |

### Class-Level Performance Breakdown (Test Split: 508 sentences)
| Class Label | Precision | Recall | F1-Score | Support (Sentences) |
| :--- | :--- | :--- | :--- | :--- |
| **Non-biased (0)** | 82.10% | 83.73% | 82.91% | 252 |
| **Biased (1)** | 83.67% | 82.03% | 82.84% | 256 |
| **Macro Average** | **82.88%** | **82.88%** | **82.87%** | **508** |

### Confusion Matrix
*(Rows denote actual labels, columns denote predicted labels: `[Non-biased, Biased]`)*

$$\begin{pmatrix} 211 & 41 \\ 46 & 210 \end{pmatrix}$$

#### Plain-Language Outcome Breakdown:
- **211 non-biased sentences correctly identified as non-biased** (True Negatives)
- **41 non-biased sentences mistakenly flagged as biased** (False Positives)
- **46 biased sentences missed by the model** (False Negatives)
- **210 biased sentences correctly detected as biased** (True Positives)

> **Evaluation Note**: *The reported test set has been examined across multiple experiments during development. These results should therefore be interpreted as exploratory rather than as a completely untouched final evaluation. Performance on new data may differ.*

---

## 3. Repository Structure

```text
news-bias-detection/
├── app/                        # FastAPI backend application
│   ├── __init__.py
│   ├── dependencies.py         # Singleton model & explainer lifecycle management
│   ├── main.py                 # API routes (/health, /predict, /explain, /metrics, /)
│   └── schemas.py              # Pydantic v2 validation models
├── data/                       # Datasets and raw annotation files
│   ├── processed/
│   │   └── newsbias_cleaned.csv
│   └── raw/
│       ├── allsides_balanced_news_headlines-texts.csv
│       ├── final_labels_SG1.xlsx
│       └── final_labels_SG2.xlsx
├── frontend/                   # React + Vite interactive user interface
│   ├── dist/                   # Production-compiled assets (served by FastAPI or Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── BiasCard.jsx            # Prediction badge & probability breakdown
│   │   │   ├── ExplanationCard.jsx     # SHAP token attribution heatmap & word ranking
│   │   │   ├── LimitationsNotice.jsx   # Academic scope disclaimer
│   │   │   ├── MetricsModal.jsx        # Benchmark test report modal (DeBERTa-v3-base)
│   │   │   └── SentenceBreakdown.jsx   # Sentence-level analysis & per-sentence SHAP trigger
│   │   ├── App.jsx             # Main dashboard application
│   │   ├── index.css           # Modern accessible responsive styling
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js          # Vite config with API proxy
├── model_deberta/              # Active production fine-tuned DeBERTa-v3-base checkpoint
│   ├── config.json             # Architecture & id2label mapping ({0: Non-biased, 1: Biased})
│   ├── model.safetensors       # DeBERTa fine-tuned weights (~737 MB)
│   ├── tokenizer.json          # Fast tokenizer vocabulary & merges
│   ├── tokenizer_config.json   # Tokenizer parameters (max_length: 256)
│   └── training_args.bin       # PyTorch training arguments binary (/content/deberta_babe_only)
├── notebooks/                  # Academic experiments & research notebooks
│   ├── babe_dataset.ipynb      # BABE dataset cleaning, training, and checkpoint export
│   ├── baseline_comparison.ipynb# Baseline ML models (Linear SVM, Logistic Regression)
│   ├── data_cleaning.ipynb     # Initial data processing
│   ├── data_exploration.ipynb  # EDA and label distribution analysis
│   ├── explainability.ipynb    # SHAP explanations, token attributions, and case studies
│   └── roberta_experiment.ipynb# Historical transformer experimentation (retained for reference)
├── results/                    # Baseline benchmark reports & figures
│   └── baseline_comparison.csv
├── src/                        # Core reusable Python library
│   ├── models/
│   │   ├── explainer.py        # Reusable SHAP Text explainer & word attribution
│   │   └── inference.py        # Reusable NewsBiasClassifier (validation, inference)
│   └── utils/
│       ├── config.py           # Centralized configuration & environment loader
│       └── text.py             # Sentence segmentation and whitespace normalization
├── tests/                      # Automated test suite
│   ├── test_api.py             # FastAPI route integration tests (TestClient)
│   └── test_inference.py       # Model loading, validation, and prediction tests
├── .env.example                # Template configuration file
├── .gitignore                  # Git rules protecting model weights, venv, and builds
├── package.json                # Root package delegating to frontend scripts
├── requirements.txt            # Python dependencies (UTF-8)
└── README.md                   # Complete documentation
```

---

## 4. Setup and Installation

### Prerequisites
- Windows 10/11 with **PowerShell**
- **Python 3.10+** (Python 3.13 tested and verified)
- **Node.js v18+** and **npm** (Node v22 tested)

### 1. Python Environment Setup
Open PowerShell and navigate to the project root:

```powershell
# Create virtual environment (if not already present)
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

---

## 5. Model Checkpoint Verification

Ensure the local model checkpoint directory `model_deberta` contains the following verified files:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`

*(Note: Model binaries `*.safetensors` are ignored by git in `.gitignore` to prevent large binary bloat in version control).*

If your model is stored in a custom path, set the `MODEL_DIR` environment variable:
```powershell
$env:MODEL_DIR = "model_deberta"
```

---

## 6. How to Run the Application

You can run the project in two convenient ways:

### Option A: Unified Full-Stack Server (Recommended for Quick Demo)
FastAPI automatically serves the pre-compiled React frontend alongside all API endpoints on a single port:

```powershell
# Ensure virtual environment is active
.\.venv\Scripts\Activate.ps1

# Start the FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Web Application URL**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Option B: Independent Backend & Frontend Dev Servers (For Development)

**Terminal 1 — FastAPI Backend:**
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Vite React Frontend:**
```powershell
cd frontend
npm run dev
```

- **Frontend Development URL**: [http://localhost:5173/](http://localhost:5173/)
- The Vite development server automatically proxies requests from `/api/*` to `http://127.0.0.1:8000/*`.

---

## 7. How to Run Tests

Run the automated test suite using `pytest`:

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/ -v
```

This verifies:
1. Model weights and tokenizer load from local checkpoint without internet access.
2. Inference on neutral and biased sample text produces valid classes and probabilities.
3. Probability distribution constraints ($P \in [0, 1]$ and $\sum P \approx 1.0$).
4. Validation on empty, whitespace-only, and malformed inputs.
5. Batch prediction functionality.
6. API `/health`, `/predict`, `/explain`, and `/metrics` routes.
7. Error response handling for invalid inputs.

---

## 8. API Endpoint Documentation

### `GET /health`
Returns system status and model readiness.

**PowerShell Example:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "model_path": "...\\model_deberta",
  "id2label": {
    "0": "Non-biased",
    "1": "Biased"
  },
  "version": "0.1.0"
}
```

---

### `POST /predict`
Performs bias classification on the submitted text.

**Request Body:**
```json
{
  "text": "The corrupt government shamelessly betrayed ordinary citizens with its disastrous policy.",
  "include_explanation": false
}
```

**PowerShell Example:**
```powershell
$body = @{
    text = "The government announced a new policy today, according to officials."
    include_explanation = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json
```

**Response:**
```json
{
  "text": "The committee met on Monday. However, corrupt officials betrayed voters with a disastrous package.",
  "predicted_label": "Biased",
  "predicted_id": 1,
  "confidence": 0.5019,
  "probabilities": {
    "Non-biased": 0.4981,
    "Biased": 0.5019
  },
  "biased_probability": 0.5019,
  "non_biased_probability": 0.4981,
  "article_summary": {
    "total_sentences": 2,
    "biased_sentences_count": 1,
    "non_biased_sentences_count": 1,
    "biased_sentences_percentage": 50.0,
    "mean_biased_probability": 0.5019,
    "max_biased_probability": 0.9969,
    "bias_intensity": "Moderate",
    "aggregation_method": "Classified as Biased: 1 of 2 sentences (50.0%) show biased framing (exceeding the 25% article threshold), with mean biased probability 50.19%."
  },
  "sentences": [
    {
      "index": 1,
      "text": "The committee met on Monday.",
      "predicted_label": "Non-biased",
      "predicted_id": 0,
      "confidence": 0.9931,
      "probabilities": { "Non-biased": 0.9931, "Biased": 0.0069 },
      "biased_probability": 0.0069,
      "non_biased_probability": 0.9931,
      "explanation": null
    },
    {
      "index": 2,
      "text": "However, corrupt officials betrayed voters with a disastrous package.",
      "predicted_label": "Biased",
      "predicted_id": 1,
      "confidence": 0.9969,
      "probabilities": { "Non-biased": 0.0031, "Biased": 0.9969 },
      "biased_probability": 0.9969,
      "non_biased_probability": 0.0031,
      "explanation": null
    }
  ],
  "explanation": null
}
```

---

### `POST /explain`
Generates SHAP word-level feature attributions for a text input.

**Request Body:**
```json
{
  "text": "The corrupt administration recklessly betrayed citizens.",
  "top_n": 5
}
```

**Response:**
```json
{
  "success": true,
  "base_value": 0.521,
  "top_words": [
    {
      "word": "corrupt",
      "score": 0.3989,
      "absolute_importance": 0.3989,
      "direction": "Pushes toward Biased"
    },
    {
      "word": "recklessly",
      "score": 0.2132,
      "absolute_importance": 0.2132,
      "direction": "Pushes toward Biased"
    },
    {
      "word": "betrayed",
      "score": 0.128,
      "absolute_importance": 0.128,
      "direction": "Pushes toward Biased"
    }
  ],
  "token_highlights": [
    { "token": "The ", "score": 0.027, "direction": "Pushes toward Biased" },
    { "token": "corrupt ", "score": 0.3989, "direction": "Pushes toward Biased" },
    { "token": "administration ", "score": -0.0249, "direction": "Pushes away from Biased" }
  ],
  "disclaimer": "SHAP feature attributions reflect which words shifted this specific model's prediction score toward or away from 'Biased'. They do not prove objective truth or publisher ideology."
}
```

---

## 9. Known Limitations & Interpretability Caveats

1. **Academic Scope**: The model is trained on labeled media sentences from the BABE dataset. It captures linguistic indicators such as loaded words, dramatic adjectives, and one-sided framing.
2. **Not a Fact Checker**: High confidence in "Non-biased" does **not** indicate that a claim is factually true; it merely indicates that the phrasing avoids overt sensationalism.
3. **No Publisher Ideology**: The classifier predicts binary framing bias and does not assign political ideology (e.g., liberal, conservative, left, right).
4. **SHAP Latency**: Computing SHAP kernel evaluations across masked token perturbations requires multiple forward passes. On CPU, explanations typically take 2–4 seconds per sentence.
5. **Contextual Subtleties**: Sarcasm, satire, and implicit bias without overt keywords may be misclassified.

---

## 10. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
