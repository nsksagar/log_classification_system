# Log Classification System

A hybrid log classification pipeline that combines Regex, Sentence Transformer embeddings (Logistic Regression), and an LLM to classify log messages into structured categories.

---

## Project Structure

```
log_classification_system/
│
├── classify.py               # Main classification pipeline
├── processor_regex.py        # Rule-based regex classifier
├── processor_bert.py         # Sentence Transformer + Logistic Regression classifier
├── processor_llm.py          # LLM-based classifier (Groq / LLaMA)
│
├── models/
│   └── log_classifier.joblib # Trained Logistic Regression model
│
├── resources/
│   ├── input.csv             # Input logs for classification
│   └── output.csv            # Classification results (generated)
│
└── training/
    ├── training.ipynb        # Model training notebook
    └── dataset/
        └── synthetic_logs.csv  # Training dataset
```

---

## Classification Pipeline

The system uses a **three-stage hybrid pipeline**:

```
Input Log
    │
    ▼
[1] LegacyCRM source? ──Yes──► LLM Classifier (Groq / LLaMA-3.3-70b)
    │
    No
    ▼
[2] Regex Match? ──Yes──► Regex Label
    │
    No
    ▼
[3] Sentence Transformer + Logistic Regression ──► Predicted Label (or "Unclassified")
```

### Stage 1 — Regex (`processor_regex.py`)
Matches structured, predictable log patterns using regular expressions.

**Labels:** `User Action`, `System Notification`

### Stage 2 — Sentence Transformer + Logistic Regression (`processor_bert.py`)
For logs that don't match any regex pattern:
- Converts log message to a vector embedding using `all-MiniLM-L6-v2`
- Feeds the embedding into a trained Logistic Regression model
- Returns `Unclassified` if the model confidence is below 50%

### Stage 3 — LLM (`processor_llm.py`)
Used exclusively for `LegacyCRM` source logs (legacy/unstructured messages):
- Sends the log to **LLaMA-3.3-70b** via Groq API
- Extracts the category from `<category>` tags in the response

**Labels:** `Workflow Error`, `Deprecation Warning`, `Unclassified`

---

## Setup

### 1. Install dependencies

```bash
pip install sentence-transformers scikit-learn joblib pandas groq python-dotenv
```

### 2. Set up environment variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run classification on a CSV file

```bash
python classify.py
```

This reads from `resources/input.csv` and writes results to `resources/output.csv`.

---

## Input Format

`input.csv` must have the following columns:

| Column        | Description                        |
|---------------|------------------------------------|
| `source`      | Log source (e.g. `LegacyCRM`, `ModernHR`) |
| `log_message` | The raw log message text           |

---

## Output Format

`output.csv` contains all original columns plus:

| Column         | Description                  |
|----------------|------------------------------|
| `target_label` | Predicted classification label |

---

## Training

The Logistic Regression model was trained in `training/training.ipynb`:

1. Loaded `synthetic_logs.csv`
2. Applied DBSCAN clustering on sentence embeddings for exploratory analysis
3. Classified easy/structured logs using regex
4. Removed `LegacyCRM` source (handled by LLM)
5. Generated sentence embeddings using `all-MiniLM-L6-v2`
6. Trained a Logistic Regression model on the remaining logs
7. Saved the model to `models/log_classifier.joblib`

---

## Notes

- The `resources/` folder must exist before running classification
- The `models/log_classifier.joblib` file must be present (train first if missing)
- Logs with model confidence below 50% are labeled `Unclassified`