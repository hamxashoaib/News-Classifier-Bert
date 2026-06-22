# 📰 News Topic Classifier Using BERT


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Objective

Fine-tune `bert-base-uncased` on the **AG News** dataset to automatically classify news headlines into **4 topic categories**:

| Label | Category | Description |
|-------|----------|-------------|
| 0 | 🌍 World | International news & geopolitics |
| 1 | ⚽ Sports | Sports events & results |
| 2 | 💼 Business | Finance, economy & markets |
| 3 | 🔬 Sci/Tech | Science and technology |

---

## 📁 Repository Structure

```News-Classifier-Bert/
│
├── News_classifier_bert.ipynb   ← Main notebook (training + evaluation)
├── app.py                             ← Streamlit deployment app
├── requirements.txt                   ← Python dependencies
└── README.md                          ← This file
```

---

## 🔬 Methodology / Approach

### 1. Dataset
- **AG News** loaded from [Hugging Face Datasets](https://huggingface.co/datasets/ag_news)
- 120,000 training samples · 7,600 test samples
- Perfectly balanced — 30,000 samples per class in training

### 2. Preprocessing
- Tokenized using `bert-base-uncased` tokenizer
- `max_length = 128` tokens (covers 95%+ of AG News texts)
- Dynamic padding via `DataCollatorWithPadding`

### 3. Model Architecture
```
bert-base-uncased (110M params)
    └── Linear(768 → 4)   ← 4-class classification head
```
- Pre-trained BERT frozen initially; full model fine-tuned end-to-end
- Transfer learning from general English text → news classification

### 4. Training Setup
| Hyperparameter | Value |
|---------------|-------|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| LR schedule | Linear warmup (200 steps) |
| Weight decay | 0.01 |

### 5. Evaluation Metrics
- **Accuracy** — overall correctness
- **Weighted F1-Score** — handles class balance, considers precision & recall per class
- Confusion matrix and per-class breakdown included

### 6. Deployment
- **Streamlit** web app (`app.py`) for live headline classification
- User types a headline → model returns predicted category + confidence score

---

## 📊 Key Results & Observations

| Metric | Score |
|--------|-------|
| Test Accuracy | **~94%** |
| Weighted F1-Score | **~0.94** |
| Best Epoch | 3 |

### Insights
1. **Transfer learning is highly efficient** — BERT achieves >90% with just 3 epochs and a subset of training data, far outperforming training from scratch.
2. **Balanced dataset = clean evaluation** — No class weighting needed; all categories perform comparably.
3. **Main confusion area**: `Business` ↔ `Sci/Tech` (e.g., tech company earnings reports appear in both).
4. **128 max token length** is the sweet spot for AG News — short texts, low compute cost, no truncation loss.
5. Fine-tuned BERT generalizes well to unseen headlines, even unconventional phrasing.

---

## 🚀 How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/hamxashoaib/News-Classifier-Bert
cd News-Classifier-Bert
pip install -r requirements.txt
```

### 2. Train the Model
```bash
jupyter notebook News_classifier_bert.ipynb
# Run all cells top to bottom
# Model saved to: ./bert-ag-news-finetuned/
```

### 3. Launch Streamlit App
```bash
streamlit run app.py
```
Open browser at `http://localhost:8501`

---

## 🛠 Tech Stack

- **Model**: `bert-base-uncased` via [Hugging Face Transformers](https://huggingface.co/transformers)
- **Dataset**: `ag_news` via [Hugging Face Datasets](https://huggingface.co/datasets)
- **Training**: HuggingFace `Trainer` API
- **Evaluation**: `scikit-learn` (accuracy, F1, confusion matrix)
- **Deployment**: `Streamlit`
- **Visualization**: `matplotlib`, `seaborn`

---

## 👨‍💻 Author

**Hamza Shoaib**  
BS Artificial Intelligence — Islamia University of Bahawalpur  

---

