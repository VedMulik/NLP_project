# Political Interview Clarity Classification (CLARITY Task)

## Team Members
- **Shikha Shashikant Masurkar**
- **Ved Sanjay Mulik**
- **Jai Damani**

---

## Project Overview
Political interviews often contain vague, indirect, or evasive responses, making it difficult to determine whether a question has been genuinely answered. This project focuses on the **CLARITY task**, which classifies interview responses into three categories:

- **Clear Reply (CR):** Directly answers the question.
- **Clear Non-Reply (CNR):** Avoids or dodges the question.
- **Ambivalent (AMB):** Partially informative but deliberately vague.

Using the QEvasion dataset, we evaluate several computational approaches, including:

- Classical machine learning  
- Embedding-based retrieval  
- Zero-shot large language models (LLMs)  
- Hybrid retrieval-augmented generation methods  

---

## Folder Structure
```
.
├── README.md                # Project documentation
├── classical.py             # TF-IDF + Logistic Regression baseline
├── classical_model.pkl      # Saved classical model
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
├── load_data.py             # Data loading and preprocessing
├── non_fine_tune.py         # Zero-shot LLaMA 3.1 8B classification
├── rag.py                   # Embedding-based retrieval classifier
├── rag_llm.py               # Hybrid RAG + LLaMA classification
```

---

## Methods

### 1. Classical Machine Learning Baseline
Uses **TF-IDF (unigrams + bigrams)** with **Logistic Regression**.

- Captures lexical patterns  
- Performs well for **Ambivalent (AMB)** responses  
- Struggles with minority **CNR** class  

---

### 2. Embedding-Based Retrieval Classification
- Uses **SentenceTransformer embeddings**  
- Nearest-neighbor retrieval + majority-vote classification  
- Improves recognition of **AMB**  
- Still challenged by label imbalance  

---

### 3. Zero-Shot LLaMA 3.1 8B
- Uses LLaMA in **next-token logit scoring mode**  
- Strong predictions for **CR** and **AMB**  
- Consistently fails to predict **CNR** due to scarcity in dataset  

---

### 4. Hybrid RAG + LLaMA
- Retrieves similar QA pairs then feeds them into LLaMA for reasoning  
- Improves **CR recall**  
- Limited gains for **CNR**, still dominated by class imbalance  

---

## Evaluation Results

| Model                                  | Weighted F1 | Macro F1 | Accuracy | Notes |
|----------------------------------------|-------------|----------|----------|-------|
| Classical TF-IDF + Logistic Regression | **0.547**   | 0.458    | 0.529    | Strong baseline; struggles with minority class |
| RAG-Style Embedding                    | 0.546       | 0.457    | 0.533    | Slight improvement on AMB |
| Zero-Shot LLaMA 3.1 8B                 | **0.573**   | 0.366    | **0.591** | Best accuracy; predicts no CNR |
| Hybrid RAG + LLaMA                     | 0.529       | 0.340    | 0.529    | Better CR recall; CNR still weak |

### Interpretation
- Classical and embedding-based models remain competitive on small datasets.  
- LLM zero-shot approaches achieve higher overall accuracy but fail on minority classes.  
- Hybrid methods help stabilize results but cannot fully overcome data imbalance.  

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Load and Preprocess Data
```bash
python load_data.py
```

### 3. Run Models

#### Classical Baseline
```bash
python classical.py --train data/train.csv --test data/test.csv
```

#### Embedding-Based Retrieval
```bash
python rag.py --train data/train.csv --test data/test.csv
```

#### Zero-Shot LLaMA
```bash
python non_fine_tune.py --test data/test.csv
```

#### Hybrid RAG + LLaMA
```bash
python rag_llm.py --train data/train.csv --test data/test.csv
```

### 4. Load Pretrained Classical Model
```python
import pickle

with open("classical_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
```

---

## Limitations
- Dataset imbalance heavily affects the minority **CNR** class.  
- LLaMA zero-shot models tend to default to **AMB** due to class distribution.  
- Models do not incorporate deeper discourse or conversational context.  
- Retrieval-based methods rely strongly on nearest-neighbor quality.  

