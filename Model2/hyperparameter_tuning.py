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
from itertools import product

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class IntentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features).to(device)
        self.labels = torch.LongTensor(labels).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class IntentClassifier(nn.Module):
    def __init__(self, input_size, num_layers, units_per_layer, dropout_rate, num_classes):
        super(IntentClassifier, self).__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_size, units_per_layer[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional layers
        for i in range(1, num_layers):
            layers.append(nn.Linear(units_per_layer[i-1], units_per_layer[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(units_per_layer[-1], num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('Cleaned_Dataset.csv')
data = data.dropna(subset=['Processed_Text', 'Label'])

# Extract features and labels
intents = data['Processed_Text']
utterances = data['Label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(intents)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    utterances, y_encoded, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    strip_accents='unicode',
    analyzer='word',
    min_df=2
)

X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_val_vect = vectorizer.transform(X_val).toarray()

# Feature selection
selector = SelectKBest(f_classif, k=min(20000, X_train_vect.shape[1]))
X_train_vect = selector.fit_transform(X_train_vect, y_train).astype('float32')
X_val_vect = selector.transform(X_val_vect).astype('float32')

# Create datasets
train_dataset = IntentDataset(X_train_vect, y_train)
val_dataset = IntentDataset(X_val_vect, y_val)

# Full hyperparameter search space
hyperparameters = {
    'num_layers': [1, 2, 3],     # 3 options
    'units': [32, 64, 128],      # 3 options
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],  # 4 options
    'learning_rate': [1e-4, 1e-3, 5e-3]    # 3 options
}

# Store results
results = []

# Hyperparameter tuning
print("\nStarting hyperparameter tuning...")
best_val_acc = 0
best_params = None
best_model_state = None

total_combinations = (len(hyperparameters['num_layers']) * 
                     len(hyperparameters['units']) * 
                     len(hyperparameters['dropout_rate']) * 
                     len(hyperparameters['learning_rate']))

print(f"Total combinations to try: {total_combinations}")
current_combination = 0

for num_layers, units, dropout_rate, lr in product(
    hyperparameters['num_layers'],
    hyperparameters['units'],
    hyperparameters['dropout_rate'],
    hyperparameters['learning_rate']
):
    current_combination += 1
    print(f"\nTrying combination {current_combination}/{total_combinations}")
    print(f"Parameters: layers={num_layers}, units={units}, dropout={dropout_rate}, lr={lr}")
    
    # Create units per layer list (same units for all layers)
    units_per_layer = [units] * num_layers
    
    # Create model
    model = IntentClassifier(
        input_size=X_train_vect.shape[1],
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        dropout_rate=dropout_rate,
        num_classes=len(label_encoder.classes_)
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training loop
    for epoch in range(20):
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
    
    # Calculate final accuracies
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    
    # Store results
    results.append({
        'num_layers': num_layers,
        'units': units,
        'dropout_rate': dropout_rate,
        'learning_rate': lr,
        'train_acc': train_acc * 100,
        'val_acc': val_acc * 100
    })
    
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Update best model if necessary
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {
            'num_layers': num_layers,
            'units': units,
            'dropout_rate': dropout_rate,
            'learning_rate': lr
        }
        best_model_state = model.state_dict().copy()

# Print best results
print("\n=== Best Configuration Found ===")
print("\nBest Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")

# Convert results to DataFrame and sort by validation accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_acc', ascending=False)

print("\n=== Top 10 Best Configurations ===")
print(results_df.head(10).to_string(index=False))

# Save best model and components
print("\nSaving best model and components...")
torch.save({
    'model_state_dict': best_model_state,
    'model_config': {
        'input_size': X_train_vect.shape[1],
        'num_classes': len(label_encoder.classes_),
        'best_params': best_params
    }
}, 'best_model.pt')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save results
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print("\nResults saved to 'hyperparameter_tuning_results.csv'")