

import torch
import torch.nn.functional as F
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from load_data import load_qevasion
from sklearn.metrics import classification_report
import tqdm


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = ""
LABELS = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]  
TOP_K = 5  


train_df, test_df = load_qevasion()
train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()


print("Computing train embeddings...")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
train_embeddings = embedder.encode(train_texts, convert_to_tensor=True, batch_size=64)

print("Computing test embeddings...")
test_embeddings = embedder.encode(test_texts, convert_to_tensor=True, batch_size=64)


print("Loading LLaMA model and tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)
model.eval()


def retrieve_top_k(test_emb, train_embs, train_texts, train_labels, k=5):
    sims = torch.matmul(train_embs, test_emb) / (
        torch.norm(train_embs, dim=1) * torch.norm(test_emb) + 1e-8
    )
    topk = torch.topk(sims, k)
    return [(train_texts[i], train_labels[i]) for i in topk.indices]

def classify_with_context(test_text, retrieved, labels=LABELS):

    context_texts = "\n".join([f"Answer: {txt} Label: {labels[label]}" for txt, label in retrieved])
    
    prompt = (
        "You are a political interview classifier. "
        "Given previous examples, classify the following answer into one of these categories:\n"
        f"1. {labels[0]}\n2. {labels[1]}\n3. {labels[2]}\n\n"
        f"Context:\n{context_texts}\n\n"
        f"Question + Answer:\n{test_text}\nLabel (choose only 1): "
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    label_ids = [tokenizer(label, add_special_tokens=False).input_ids for label in labels]
    label_logits = torch.tensor([logits[i[0]] for i in label_ids]).to(DEVICE)
    probs = F.softmax(label_logits, dim=0)
    pred_idx = torch.argmax(probs).item()
    return pred_idx


preds = []
for i, (text, test_emb) in enumerate(tqdm.tqdm(zip(test_texts, test_embeddings), total=len(test_texts))):
    retrieved = retrieve_top_k(test_emb, train_embeddings, train_texts, train_labels, k=TOP_K)
    pred = classify_with_context(text, retrieved)
    preds.append(pred)


print("\n=== RAG + LLaMA Hybrid Classification Report ===")
print(classification_report(true_labels, preds, target_names=LABELS, digits=4))
