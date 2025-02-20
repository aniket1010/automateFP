import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "Dataset.xlsx"
df = pd.read_excel(file_path)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Intent'])
num_labels = len(label_encoder.classes_)

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Utterance'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

train_dataset = IntentDataset(train_texts, train_labels)
val_dataset = IntentDataset(val_texts, val_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save the trained model
model.save_pretrained("./intent_classifier_model")
tokenizer.save_pretrained("./intent_classifier_model")

# Save label mapping
import json
with open("./intent_classifier_model/label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)
