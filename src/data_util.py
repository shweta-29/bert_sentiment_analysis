# Scripts for loading and processing data
import pandas as pd
from transformers import BertTokenizer
import torch

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    # Basic preprocessing: remove unwanted characters, etc.
    return text.strip()

def prepare_data(data):
    data['Text'] = data['Text'].apply(preprocess_text)
    return data

def tokenize_data(data, tokenizer):
    tokens = tokenizer(data['Text'].tolist(),truncation=True,padding=True,max_length=128,return_tensors='pt')
    return tokens

if __name__ == "__main__":
    file_path = "data/raw/BERT_synthetic_input_data.csv"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data = load_data(file_path)
    prepared_data = prepare_data(data)
    tokenized_data = tokenize_data(prepared_data,tokenizer)
    print("Data Preparation Complete!")
    print(f"Tokenized Input IDs: {tokenized_data['input_ids']}")    
    # Save the tokenized data to a file
    torch.save(tokenized_data, 'tokenized_data.pt')
    print("Tokenized data saved to 'tokenized_data.pt'.")
