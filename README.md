Project Overview

Political interviews often contain vague or indirect answers, making it challenging to determine whether a question has been directly addressed. The CLARITY task formalizes this problem by categorizing responses into:

Clear Reply – direct answers to the question

Clear Non-Reply – evasive or non-answers

Ambivalent – partially informative but deliberately vague responses

This project explores computational approaches for automatically classifying political interview answers using the QEvasion dataset. We evaluate:

Classical ML Baseline: TF-IDF features + Logistic Regression

Embedding-Based Retrieval: Semantic similarity with SentenceTransformers embeddings and nearest-neighbor label propagation

Zero-Shot LLaMA 3.1 8B: Instruction-tuned LLM with next-token logit scoring for classification

Hybrid RAG + LLaMA: Combining retrieval-based context with LLM reasoning

Repository Structure
├── data/                  # Dataset files (QEvasion)
├── notebooks/             # Jupyter notebooks for experiments and analysis
├── src/                   # Source code for model training and evaluation
│   ├── baseline.py        # TF-IDF + Logistic Regression
│   ├── embedding_rag.py   # Embedding-based retrieval classifier
│   ├── llama_zero_shot.py # Zero-shot LLaMA evaluation
│   └── hybrid_rag_llama.py# Hybrid RAG + LLaMA approach
├── results/               # Model outputs and performance reports
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

Setup Instructions

Clone the repository

git clone https://github.com/your-username/clarity-political-nlp.git
cd clarity-political-nlp


Install dependencies

pip install -r requirements.txt


Data

Place QEvasion dataset files in the data/ directory.

Ensure files are properly formatted as question–answer pairs with corresponding labels.

Run Models

Classical Baseline:

python src/baseline.py --train data/train.csv --test data/test.csv


Embedding-Based Retrieval:

python src/embedding_rag.py --train data/train.csv --test data/test.csv


Zero-Shot LLaMA:

python src/llama_zero_shot.py --test data/test.csv


Hybrid RAG + LLaMA:

python src/hybrid_rag_llama.py --train data/train.csv --test data/test.csv

Evaluation Metrics

We report model performance using:

Precision, Recall, F1-Score for each class

Macro F1 – averages performance across all classes

Weighted F1 – accounts for class imbalance

Accuracy – overall proportion of correct predictions

Key Findings
Model	Macro F1	Weighted F1	Accuracy	Notes
TF-IDF + Logistic Regression	0.458	0.547	0.529	Good on majority class; struggles with Clear Non-Reply
RAG Embedding Classifier	0.457	0.546	0.532	Slight improvement for minority class; semantic retrieval helps
Zero-Shot LLaMA 3.1 8B	0.366	0.573	0.591	Strong overall accuracy; fails to predict minority class
Hybrid RAG + LLaMA	0.340	0.529	0.529	Improves recall for Clear Reply; limited effect on Clear Non-Reply

Insights:

Classical and embedding-based methods remain competitive for small datasets.

Zero-shot LLMs excel in overall accuracy but struggle with minority classes due to imbalance.

Hybrid retrieval provides contextual grounding but cannot fully compensate for underrepresented classes.

Limitations

Severe class imbalance in QEvasion dataset

Zero-shot LLMs may default to majority class without fine-tuning

Contextual information beyond question-answer pairs is not considered

Retrieval methods are dependent on embedding quality and coverage

Future Work

Fine-tuning LLMs with task-specific supervision

Incorporating conversational context and discourse-level features

Using data augmentation or rebalancing for minority classes

Exploring contrastive learning or specialized architectures for subtle evasive answers

Acknowledgements

CSCI 5832 (Natural Language Processing) – University of Colorado Boulder for guidance and support

QEvasion dataset creators for providing labeled political interviews

Developers of Hugging Face Transformers, SentenceTransformers, and Python ML ecosystem
