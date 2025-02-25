import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from itertools import product

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(num_layers, units_per_layer, dropout_rate, input_shape, num_classes, learning_rate):
    """Create model with specified hyperparameters"""
    model = Sequential()
    
    # First layer
    model.add(Dense(units_per_layer[0], activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional layers
    for i in range(1, num_layers):
        model.add(Dense(units_per_layer[i], activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load and preprocess data
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

# Hyperparameter search space
hyperparameters = {
    'num_layers': [1, 2, 3],  # Number of layers to try
    'units': [32, 64, 128],   # Units per layer
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],  # Dropout rates
    'learning_rate': [1e-4, 1e-3, 5e-3]    # Learning rates
}

# Store results
results = []

# Hyperparameter tuning
print("Starting hyperparameter tuning...")
best_val_acc = 0
best_params = None
best_model = None

for num_layers, units, dropout_rate, lr in product(
    hyperparameters['num_layers'],
    hyperparameters['units'],
    hyperparameters['dropout_rate'],
    hyperparameters['learning_rate']
):
    # Create units per layer list
    units_per_layer = [units] * num_layers
    
    print(f"\nTrying: layers={num_layers}, units={units}, dropout={dropout_rate}, lr={lr}")
    
    # Create and train model
    model = create_model(
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        dropout_rate=dropout_rate,
        input_shape=(X_train_vect.shape[1],),
        num_classes=len(label_encoder.classes_),
        learning_rate=lr
    )
    
    # Train model
    history = model.fit(
        X_train_vect,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val_vect, y_val),
        verbose=0
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train_vect, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val_vect, y_val, verbose=0)
    
    # Store results
    results.append({
        'num_layers': num_layers,
        'units': units,
        'dropout_rate': dropout_rate,
        'learning_rate': lr,
        'train_acc': train_acc,
        'val_acc': val_acc
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
        best_model = model

# Print best results
print("\nBest Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")

# Save best model and components
best_model.save('best_model.keras')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save results to CSV for analysis
results_df = pd.DataFrame(results)
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print("\nResults saved to 'hyperparameter_tuning_results.csv'")