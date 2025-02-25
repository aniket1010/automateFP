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

# Load the dataset
data = pd.read_csv('Cleaned_Dataset.csv')

# Ensure the required columns exist
required_columns = {'Processed_Text', 'Label'}
if not required_columns.issubset(data.columns):
    raise KeyError(f"Missing required columns: {required_columns - set(data.columns)}")

# Drop missing values
data = data.dropna(subset=['Processed_Text', 'Label'])

# Extract intents and utterances
intents = data['Processed_Text']  # Intent names
utterances = data['Label']  # Corresponding utterances

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(intents)

# Handle cases where some intents have less than 2 samples
if np.min(np.bincount(y_encoded)) < 2:
    stratify_option = None  # Disable stratification if any intent has fewer than 2 samples
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

# Define Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_vect.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_vect, y_train, epochs=20, batch_size=32, validation_data=(X_val_vect, y_val))

# Evaluate model
train_loss, train_acc = model.evaluate(X_train_vect, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val_vect, y_val, verbose=0)

print(f'Training Accuracy: {train_acc * 100:.2f}%')
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

# Save the trained model and preprocessing objects
model.save('trained_model.h5')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and preprocessing objects saved successfully.")
