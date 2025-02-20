import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments , EarlyStoppingCallback
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "./Balanced_dataset.xlsx"
df = pd.read_excel(file_path)

# Normalize utterances
df["Utterance"] = df["Utterance"].str.lower().str.strip()
df = df.drop_duplicates(subset=["Utterance"])  # Remove duplicates
df["word_count"] = df["Utterance"].apply(lambda x: len(str(x).split()))
df = df[df["word_count"] >= 3]  # Remove very short utterances

# Encode intent labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Intent"])
num_labels = len(label_encoder.classes_)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Utterance"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Load DeBERTa tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert dataset to Hugging Face format
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)

train_dataset = HFDataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = HFDataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(lambda x: tokenize_function(x["text"]), batched=True)
val_dataset = val_dataset.map(lambda x: tokenize_function(x["text"]), batched=True)

# Load DeBERTa-v3-Base model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Enable PyTorch 2.0 optimizations (Speeds up training)
torch.compile(model)

# Define training arguments (optimized for RTX 4050)
training_args = TrainingArguments(
    output_dir="./results2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-6,  # Lower LR for stability
    per_device_train_batch_size=32,  # Balanced batch size for VRAM usage -> (changed from 16 prev to 32 , it was slow )
    per_device_eval_batch_size=32,
    num_train_epochs=10,  # Train longer for better accuracy
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,  # Mixed precision to save VRAM
    gradient_accumulation_steps=1,  # reduces computation time per step but requires more VRAM 
    logging_steps=500,
    report_to="none",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./deberta_base_intent_model")
tokenizer.save_pretrained("./deberta_base_intent_model")

# Save label encoder mapping
import json
with open("./deberta_base_intent_model/label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

print("✅ Training complete! Model saved at './deberta_base_intent_model'.")
