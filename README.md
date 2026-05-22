# LLM Style Analysis

A stylometric study of whether Large Language Models have a recognizable "voice" — and whether a classifier can tell them apart purely from linguistic features.

**Course:** Natural Language Processing — Prof. Alfio Ferrara, Università degli Studi di Milano  
**Project ID:** P13 — The Aesthetics of Generation

---

## Overview

This project investigates the stylistic fingerprints of five major LLMs by generating responses to the same set of prompts across six text genres, extracting quantitative linguistic features, and training a Random Forest classifier to attribute authorship to the correct model.

**Models compared:**
- GPT-4o-mini (OpenAI)
- Claude Haiku (Anthropic)
- Gemini 2.5 Flash (Google)
- Qwen 2.5 7B (Alibaba, via HuggingFace)
- Llama 3.1 8B (Meta, via HuggingFace)

**Genres:** explanation, opinion, comparison, narration, argumentation, dialogue

**Dataset:** ~600 responses (5 models × ~120 prompts, balanced across genres)

**Features extracted (14):** average sentence length, punctuation frequency, POS-tag ratios (noun, verb, adjective, adverb), type-token ratio, passive/active voice ratio, tense distribution, first-person pronoun usage, pragmatic markers (contradiction, agreement), via spaCy + [StyloMetrix](https://github.com/ZILiAT-NASK/StyloMetrix)

**Classifier:** Random Forest with GridSearchCV + 5-fold stratified cross-validation; SHAP values for interpretability

---

## Results

| Metric | Value |
|--------|-------|
| Cross-validated accuracy | ~54% (±3%) |
| Train accuracy | ~67% |
| Chance baseline | 20% (5-class) |

**Accuracy by genre:**

| Genre | Accuracy |
|-------|----------|
| comparison | 59% |
| argumentation | 57% |
| dialogue | 56% |
| opinion | 54% |
| narration | 52% |
| explanation | 44% |

SHAP analysis reveals that type-token ratio, passive/active voice, and pragmatic markers are the most discriminating features. The classifier struggles most with explanation texts, where models converge on a neutral, informative register.

---

## Repository Structure

```
nlp_project/
├── data/
│   ├── prompts.xlsx              # Input prompts (id_prompt, category, text)
│   ├── responses.xlsx            # LLM responses (prompt_id, category, prompt_text, model, response)
│   ├── sm_numeric_features.xlsx  # Extracted stylometric features (14 columns + metadata)
│   ├── classifier_results.pkl    # Serialized predictions, feature importance, labels
│   └── shap_values.npy           # SHAP values array (samples × features × classes)
├── figures/                      # Generated visualizations (local, not tracked in git)
├── notebooks/
│   └── 03_analysis.ipynb         # Full pipeline demo with outputs — start here
├── src/
│   ├── model_call.py             # API wrappers for all 5 LLMs
│   ├── data_collection.py        # Prompt → responses pipeline
│   ├── feature_extraction.py     # spaCy + StyloMetrix feature extraction
│   ├── analysis.py               # N-gram frequency analysis
│   ├── classifier.py             # Random Forest training, SHAP computation
│   └── visualization.py          # PCA, t-SNE, SHAP plots, confusion matrix
├── .env                          # API keys — NOT committed, create manually (see below)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

Python **3.9 or higher** is required.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy English model

`feature_extraction.py` requires the `en_core_web_sm` model, which is not installed automatically:

```bash
python -m spacy download en_core_web_sm
```

### 3. Create the `.env` file

Create a file named `.env` in the project root with your API keys:

```
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HF_API_KEY=your_huggingface_key_here
```

> The `.env` file is gitignored and never committed. Each key is loaded via `python-dotenv`.

---

## Reproducing Results

The steps below correspond to the pipeline stages. **Steps 1 and 2 require API access and take significant time.** If you only want to reproduce the classification and visualization results, the pre-generated data files in `data/` are already included — skip directly to Step 3.

### Step 1 — Data Collection (requires API keys)

Generates responses from all 5 models for each prompt. Must be run from inside `src/` because of a local import dependency:

```bash
cd src
python data_collection.py
```

Output: `data/responses.xlsx`

### Step 2 — Feature Extraction

Extracts stylometric features from all responses. Can be run from the project root:

```bash
python src/feature_extraction.py
```

Output: `data/sm_numeric_features.xlsx`

> This step requires both `spacy` (`en_core_web_sm`) and `stylo-metrix` to be installed.

### Step 3 — Classification

Trains the Random Forest classifier with GridSearchCV, computes SHAP values, and saves results:

```bash
python src/classifier.py
```

Outputs: `data/classifier_results.pkl`, `data/shap_values.npy`

### Step 4 — Visualization and Analysis (recommended: Jupyter Notebook)

Open the notebook for the full analysis with all visualizations and printed results:

```bash
jupyter notebook notebooks/03_analysis.ipynb
```

Alternatively, run `python src/visualization.py` in an interactive environment with a display (figures are shown via `plt.show()` and not saved automatically).

---

## AI Usage Disclaimer

Parts of this project were developed with the assistance of Claude (Anthropic) and ChatGPT (OpenAI). AI assistance was used for:
- Structuring methodological workflows and pipeline design
- Drafting and refining descriptive text
- Debugging and reviewing code

All AI-generated content has been reviewed, edited, and validated. Full responsibility for the project's structure, reasoning, and experimental choices rests with the author.
