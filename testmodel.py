from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./docsis_bert_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to make a prediction
def predict_log(log_text):
    inputs = tokenizer(log_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class

# Sample DOCSIS log entries to test
sample_logs = [
    "Received MDD message",
    "Channel Bonding Value Exceeded",
    "Upstream Channel Lock Failed"
]

# Testing the model on sample logs
for log in sample_logs:
    prediction = predict_log(log)
    print(f'Log: "{log}" Predicted Class: {prediction}')




