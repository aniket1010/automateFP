import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model and preprocessing objects
model = load_model('trained_model.h5')
vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('selector.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def predict_intent(user_input):
    """Predict the intent of a given user input."""
    # Vectorize input text
    input_vect = vectorizer.transform([user_input]).toarray()
    input_vect = selector.transform(input_vect).astype('float32')
    
    # Predict intent
    predictions = model.predict(input_vect)
    predicted_label = np.argmax(predictions, axis=1)
    
    # Decode intent
    intent = label_encoder.inverse_transform(predicted_label)[0]
    
    return intent

if __name__ == "__main__":
    while True:
        user_input = input("Enter a query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predicted_intent = predict_intent(user_input)
        print(f"Predicted Intent: {predicted_intent}")
