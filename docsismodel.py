from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your DOCSIS logs dataset
data = pd.read_csv('docsis_logs.csv')
texts = data['log_text']
labels = data['label']

# Convert labels to integers (assuming 'info', 'success', 'warning', 'error')
label_mapping = {'info': 0, 'success': 1, 'warning': 2, 'error': 3}
labels = labels.map(label_mapping).astype(int)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(inputs.input_ids, labels, test_size=0.2)

# Ensure indices match
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# Dataset class
class DOCSISDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = DOCSISDataset({'input_ids': train_texts, 'attention_mask': train_texts.ne(tokenizer.pad_token_id)}, train_labels)
val_dataset = DOCSISDataset({'input_ids': val_texts, 'attention_mask': val_texts.ne(tokenizer.pad_token_id)}, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained('./docsis_bert_model')

