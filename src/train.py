# Model training script
import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import pandas as pd

# Step 1: Load tokenised data
tokenised_data = torch.load('tokenized_data.pt', weights_only=False)
labels = pd.read_csv('data/raw/BERT_synthetic_input_data.csv')['Sentiment'].values
labels = torch.tensor(labels)

# Step 2: Create DataLoader
dataset = TensorDataset(tokenised_data['input_ids'],tokenised_data['attention_mask'], labels)
loader = DataLoader(dataset,sampler=RandomSampler(dataset), batch_size=8)

# Step 3: Load BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Step 4: Set up optimiser
optimiser = AdamW(model.parameters(),lr=5e-6)

# Step 6: Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
for batch in loader:
    b_input_ids, b_attention_mask, b_labels = [x.to(device) for x in batch]
    optimiser.zero_grad()
    outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
    loss = outputs.loss
    loss.backward()
    optimiser.step()
    print(f"Batch loss: {loss.item()}")

# Step 6: Save the trained model
model.save_pretrained("models/bert_sentiment_model")
print("Model trained and saved!")
