
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from load_data import load_qevasion


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 3  

train_df, test_df = load_qevasion()
train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()


embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

print("Computing training embeddings...")
train_embeddings = embedder.encode(train_texts, convert_to_tensor=True, device=DEVICE)
print("Computing test embeddings...")
test_embeddings = embedder.encode(test_texts, convert_to_tensor=True, device=DEVICE)


preds = []
for test_emb in test_embeddings:
    sims = torch.nn.functional.cosine_similarity(
        train_embeddings, test_emb.unsqueeze(0).expand_as(train_embeddings)
    )
    nn_idx = int(torch.argmax(sims))
    preds.append(train_labels[nn_idx])


LABELS = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]
print("\n=== RAG-style Embedding Classification Report ===")
print(classification_report(true_labels, preds, target_names=LABELS, digits=4))
