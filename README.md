# News Bias Detection

An academic AI/ML project and full-stack system for detecting bias in news text using **five trained classifiers** and a **Weighted Soft-Voting Ensemble** with **SHAP word-level explainability** and a React research dashboard.

---

## 1. System Overview & Architecture

News media framing, word choice, and tonal sentiment frequently introduce subjective bias into reporting. This project provides an end-to-end machine learning system connecting **five individual classifiers** with an empirical **Weighted Soft-Voting Ensemble**:

1. **TF-IDF + Linear SVM** (Calibrated with Platt scaling)
2. **TF-IDF + Logistic Regression** (L2-regularized linear baseline)
3. **TF-IDF + Multinomial Naive Bayes** (Frequency baseline)
4. **Fine-Tuned RoBERTa** (`FacebookAI/roberta-base`)
5. **Fine-Tuned DeBERTa** (`microsoft/deberta-v3-base`)
6. **Weighted Soft-Voting Ensemble** (Threshold: 0.40)

### Weighted Soft-Voting Ensemble Formula
Continuous probabilities from all five models are combined using empirical validation weights:

$$P_{\text{ensemble}}(\text{biased}) = \frac{1.0 \times P_{\text{SVM}} + 1.0 \times P_{\text{LR}} + 1.0 \times P_{\text{NB}} + 2.0 \times P_{\text{RoBERTa}} + 2.0 \times P_{\text{DeBERTa}}}{7.0}$$

- **Decision Rule**: Predict **Biased** if $P_{\text{ensemble}}(\text{biased}) \ge 0.40$; otherwise predict **Non-biased**.
- **Recall Optimization**: The 0.40 threshold boosts biased-class recall to **87.11%** (+5.08% improvement over the best individual model).

---

## 2. Authoritative Benchmark Evaluation Results

All models were evaluated on the held-out article-disjoint test split (**508 sentences**, 252 Non-biased, 256 Biased) from the BABE dataset:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Biased Precision | Biased Recall | Biased F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeBERTa** | **0.8287** | **0.8288** | **0.8288** | **0.8287** | **0.8367** | 0.8203 | 0.8284 |
| **Weighted Soft Voting** | 0.8268 | **0.8293** | 0.8264 | 0.8263 | 0.8022 | **0.8711** | **0.8352** |
| **RoBERTa** | 0.8209 | 0.8212 | 0.8210 | 0.8208 | 0.8340 | 0.8047 | 0.8191 |
| **Logistic Regression** | 0.7087 | 0.7097 | 0.7089 | 0.7084 | 0.7269 | 0.6758 | 0.7004 |
| **Linear SVM** | 0.7028 | 0.7072 | 0.7033 | 0.7015 | 0.7397 | 0.6328 | 0.6821 |
| **Naive Bayes** | 0.6870 | 0.6896 | 0.6874 | 0.6862 | 0.7137 | 0.6328 | 0.6708 |

### Confusion Matrices (Test Split: 508 sentences)
*(Rows: Actual `[Non-biased, Biased]`, Columns: Predicted `[Non-biased, Biased]`)*

#### 1. Weighted Soft-Voting Ensemble Matrix:
$$\begin{pmatrix} 197 & 55 \\ 33 & 223 \end{pmatrix}$$

- **197** non-biased sentences correctly identified as non-biased
- **55** non-biased sentences mistakenly flagged as biased
- **33** biased sentences missed by the ensemble
- **223** biased sentences correctly detected as biased

#### 2. DeBERTa-v3-base Matrix:
$$\begin{pmatrix} 211 & 41 \\ 46 & 210 \end{pmatrix}$$

- **211** non-biased sentences correctly identified as non-biased
- **41** non-biased sentences mistakenly flagged as biased
- **46** biased sentences missed by the model
- **210** biased sentences correctly detected as biased

#### 3. RoBERTa-base Matrix:
$$\begin{pmatrix} 211 & 41 \\ 50 & 206 \end{pmatrix}$$

- **211** non-biased sentences correctly identified as non-biased
- **41** non-biased sentences mistakenly flagged as biased
- **50** biased sentences missed by the model
- **206** biased sentences correctly detected as biased

> **Evaluation Note**: *The reported test set has been examined across multiple experiments during development. These results should therefore be interpreted as exploratory rather than as a completely untouched final evaluation. Performance on new data may differ.*

---

## 3. Model Artifact Placement

Model artifacts are placed under `models/`:

```text
models/
├── linear_svm.joblib           # CalibratedClassifierCV wrapping TfidfVectorizer + LinearSVC
├── logistic_regression.joblib  # Pipeline(TfidfVectorizer, LogisticRegression)
├── naive_bayes.joblib          # Pipeline(TfidfVectorizer, MultinomialNB)
├── ensemble_config.json        # Weights and threshold configuration
├── roberta/                    # Fine-tuned RoBERTa checkpoint
│   ├── config.json
│   ├── model.safetensors       # RoBERTa weights (~498 MB)
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── deberta/                    # Fine-tuned DeBERTa checkpoint
    ├── config.json
    ├── model.safetensors       # DeBERTa weights (~737 MB)
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 4. Setup and Installation

### Prerequisites
- Windows 10/11 with **PowerShell**
- **Python 3.10+** (Python 3.13 verified)
- **Node.js v18+** and **npm**

### Step 1: Clone and Set Up Python Virtual Environment
```powershell
# Navigate to project directory
cd "c:\Users\ADMIN\Desktop\News Bias Detection\news-bias-detection"

# Activate existing virtual environment (or create one: python -m venv .venv)
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

### Step 3: Build Frontend Production Assets
```powershell
npm run build
```

---

## 5. How to Run the Application in VS Code

You can run the application in two ways:

### Option A: Unified Full-Stack Server (Single Port 8000)
FastAPI automatically serves the compiled React application alongside all API endpoints:

```powershell
# Ensure virtual environment is active
.\.venv\Scripts\Activate.ps1

# Start server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Option B: Separate Backend & Frontend Dev Servers (With Hot Reloading)

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

- **Frontend Dev URL**: [http://localhost:5173/](http://localhost:5173/)
- Requests to `/api/*` are automatically proxied to `http://127.0.0.1:8000/*`.

---

## 6. Running Automated Tests

Run the complete test suite with `pytest`:

```powershell
.\.venv\Scripts\pytest tests/ -v
```

This verifies:
1. All 5 models load into memory.
2. Individual predictions from Linear SVM, Logistic Regression, Naive Bayes, RoBERTa, and DeBERTa.
3. Weighted soft-voting formula calculation and 0.40 threshold behavior.
4. API routes: `/api/health`, `/api/models`, `/api/predict`, `/api/metrics`, `/api/confusion-matrices`, and `/api/comparison-csv`.
5. Edge cases, sentence segmentation, whitespace normalization, and SHAP explainability.

---

## 7. Backend API Documentation

### `GET /api/health`
Returns system readiness and count of loaded models.

### `GET /api/models`
Returns detailed metadata, types, and weights for all 5 models and the ensemble.

### `POST /api/predict`
Executes multi-model inference and soft-voting aggregation.

**Sample Request:**
```json
{
  "text": "The corrupt administration recklessly betrayed citizens with its disastrous decision.",
  "include_explanation": false
}
```

**Sample Response:**
```json
{
  "input_text": "The corrupt administration recklessly betrayed citizens with its disastrous decision.",
  "individual_predictions": {
    "linear_svm": {
      "label": "Biased",
      "probability_biased": 0.6087,
      "probability_non_biased": 0.3913,
      "confidence": 0.6087,
      "model_name": "Linear SVM",
      "model_type": "TF-IDF + Linear SVM (Calibrated)"
    },
    "logistic_regression": {
      "label": "Biased",
      "probability_biased": 0.5939,
      "probability_non_biased": 0.4061,
      "confidence": 0.5939,
      "model_name": "Logistic Regression",
      "model_type": "TF-IDF + Logistic Regression"
    },
    "naive_bayes": {
      "label": "Biased",
      "probability_biased": 0.9106,
      "probability_non_biased": 0.0894,
      "confidence": 0.9106,
      "model_name": "Naive Bayes",
      "model_type": "TF-IDF + Multinomial Naive Bayes"
    },
    "roberta": {
      "label": "Biased",
      "probability_biased": 0.9969,
      "probability_non_biased": 0.0031,
      "confidence": 0.9969,
      "model_name": "RoBERTa",
      "model_type": "Fine-tuned Transformer (FacebookAI/roberta-base)"
    },
    "deberta": {
      "label": "Biased",
      "probability_biased": 0.9749,
      "probability_non_biased": 0.0251,
      "confidence": 0.9749,
      "model_name": "DeBERTa",
      "model_type": "Fine-tuned Transformer (microsoft/deberta-v3-base)"
    }
  },
  "ensemble": {
    "label": "Biased",
    "probability_biased": 0.8653,
    "probability_non_biased": 0.1347,
    "confidence": 0.8653,
    "threshold": 0.40,
    "weights": {
      "linear_svm": 1.0,
      "logistic_regression": 1.0,
      "naive_bayes": 1.0,
      "roberta": 2.0,
      "deberta": 2.0
    }
  }
}
```

### `GET /api/metrics`
Returns recorded historical performance metrics across all models and the ensemble.

### `GET /api/confusion-matrices`
Returns the available confusion matrices and outcomes.

### `GET /api/comparison-csv`
Downloads the model performance comparison table as CSV.

---

## 8. Academic Scope & Limitations

1. **Linguistic Framing vs. Factual Truth**: The system analyzes subjective word choices, emotional intensity, and framing. A sentence predicted as "Non-biased" is not guaranteed to be factually accurate.
2. **No Ideological Bias**: The system performs binary classification (Biased vs. Non-biased) and does not classify political stance (e.g. Left/Right).
3. **No Publisher-Wide Generalization**: Scores apply solely to the submitted text excerpt.
4. **Exploratory Evaluation**: Results on the 508 test sentences reflect iterative development and should be interpreted as exploratory benchmarks.
