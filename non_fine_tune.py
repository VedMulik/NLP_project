
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from load_data import load_qevasion  
from sklearn.metrics import classification_report
import tqdm


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = ""  
LABELS = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]  


train_df, test_df = load_qevasion()

test_texts = (test_df["question"] + " " + test_df["text"]).tolist()
true_labels = test_df["label"].tolist()


tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)
model.eval()


def classify_answer(text, labels=LABELS):
    """
    Zero-shot classification using next-token logits.
    Prompts model to choose from given label set.
    """
    prompt = (
        "Classify the following interview answer into one of these categories:\n"
        f"{', '.join(labels)}.\n\n"
        f"Answer:\n{text}\nLabel: "
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
for text in tqdm.tqdm(test_texts, desc="Zero-shot classification"):
    preds.append(classify_answer(text))


print("\n=== Zero-Shot LLaMA 3.1 8B Classification Report ===")
print(classification_report(true_labels, preds, target_names=LABELS, digits=4))
