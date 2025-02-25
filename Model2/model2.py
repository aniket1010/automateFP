import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Custom Dataset class
class IntentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features).to(device)
        self.labels = torch.LongTensor(labels).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural Network Model - Modified for best configuration
class IntentClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IntentClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  # Single layer with 128 units
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load the dataset
data = pd.read_csv('Cleaned_Dataset.csv')

# Ensure the required columns exist
required_columns = {'Processed_Text', 'Label'}
if not required_columns.issubset(data.columns):
    raise KeyError(f"Missing required columns: {required_columns - set(data.columns)}")

# Drop missing values
data = data.dropna(subset=['Processed_Text', 'Label'])

# Extract intents and utterances
intents = data['Processed_Text']
utterances = data['Label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(intents)

# Handle cases where some intents have less than 2 samples
if np.min(np.bincount(y_encoded)) < 2:
    stratify_option = None
else:
    stratify_option = y_encoded

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    utterances, y_encoded, test_size=0.2, random_state=42, stratify=stratify_option
)

# N-gram vectorization parameters
NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2

# Vectorization
vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE,
    dtype=np.float32,
    strip_accents='unicode',
    decode_error='replace',
    analyzer=TOKEN_MODE,
    min_df=MIN_DOCUMENT_FREQUENCY
)
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_val_vect = vectorizer.transform(X_val).toarray()

# Feature selection
selector = SelectKBest(f_classif, k=min(TOP_K, X_train_vect.shape[1]))
selector.fit(X_train_vect, y_train)
X_train_vect = selector.transform(X_train_vect).astype('float32')
X_val_vect = selector.transform(X_val_vect).astype('float32')

# Create data loaders
train_dataset = IntentDataset(X_train_vect, y_train)
val_dataset = IntentDataset(X_val_vect, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model and move to GPU
model = IntentClassifier(X_train_vect.shape[1], len(label_encoder.classes_))
model = model.to(device)

# Loss and optimizer - Using the best learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Best learning rate

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    # Print progress
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Accuracy: {train_acc * 100:.2f}%')
        print(f'Validation Accuracy: {val_acc * 100:.2f}%')

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(train_dataset.features)
    _, predicted = outputs.max(1)
    train_acc = predicted.eq(train_dataset.labels).sum().item() / len(train_dataset)
    
    outputs = model(val_dataset.features)
    _, predicted = outputs.max(1)
    val_acc = predicted.eq(val_dataset.labels).sum().item() / len(val_dataset)

print(f'\nFinal Results:')
print(f'Training Accuracy: {train_acc * 100:.2f}%')
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

# Save model and preprocessing objects
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': X_train_vect.shape[1],
        'num_classes': len(label_encoder.classes_)
    }
}, 'intent_classifier.pt')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and preprocessing objects saved successfully.")