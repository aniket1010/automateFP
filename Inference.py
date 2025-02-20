import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load the trained model and tokenizer
model_path = "./deberta_base_intent_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load label mapping (intent names)
with open(f"{model_path}/label_encoder.json", "r") as f:
    intent_labels = json.load(f)

# Move model to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_intent(user_utterance):
    """Predict the intent of a given user utterance."""
    inputs = tokenizer(user_utterance, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract predicted label (highest probability)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # Return the corresponding intent name
    return intent_labels[predicted_label]

# Example usage
user_input = "please search for an account"
predicted_intent = predict_intent(user_input)
print(f"Predicted Intent: {predicted_intent}")

