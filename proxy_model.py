import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class TextDataset(Dataset):
    """Custom Dataset for text classification."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {label: i for i, label in enumerate(np.unique(labels))}
        self.id_map = {i: label for label, i in self.label_map.items()}


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

class ProxyModel:
    """DistilBERT based Proxy Model for quick evaluation."""

    def __init__(self, num_labels, model_path='./pretrain_models/distilbert-base-cased'): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.model.to(self.device)
        self.label_encoder = LabelEncoder()
        self.max_len = 256 

    def prepare_data(self, df, fit_encoder=True):
        """Prepares data for the model, fits label encoder if needed."""
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        if fit_encoder:
            self.label_encoder.fit(labels)
            self.num_labels = len(self.label_encoder.classes_)
            self.model = DistilBertForSequenceClassification.from_pretrained('./pretrain_models/distilbert-base-cased', num_labels=self.num_labels)
            self.model.to(self.device)
        encoded_labels = self.label_encoder.transform(labels)
        return texts, encoded_labels

    def train(self, train_df, val_df, epochs=5, batch_size=16, learning_rate=2e-5):
        """Trains or fine-tunes the proxy model."""

        train_texts, train_labels = self.prepare_data(train_df, fit_encoder=False) 
        val_texts, val_labels = self.prepare_data(val_df, fit_encoder=False)

        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_len)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        best_val_macro_f1 = 0
        early_stopping_patience = 3 
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_loader)
            val_results = self.evaluate(val_loader)
            val_macro_f1 = val_results['macro_f1']
            val_accuracy = val_results['accuracy']

            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Macro-F1: {val_macro_f1:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_proxy_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                print("Early stopping!")
                break


    def evaluate(self, data_loader):
        """Evaluates the proxy model."""

        self.model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)

                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())

        predictions = self.label_encoder.inverse_transform(predictions)
        true_labels = self.label_encoder.inverse_transform(true_labels)

        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(true_labels, predictions)

        return {'macro_f1': macro_f1, 'accuracy': accuracy, 'predictions': predictions, 'true_labels': true_labels}

    def predict(self, texts):
        """Predicts labels and confidences for texts."""
        
        if not isinstance(texts, list):
            texts = [texts]

        self.model.eval()
        input_encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        dataset = TextDataset(texts, [''] * len(texts), self.tokenizer, self.max_len) # Dummy labels
        dataset.label_map = self.label_encoder.transform(self.label_encoder.classes_) # Use fitted encoder maps
        dataset.id_map = {i: label for i, label in enumerate(self.label_encoder.classes_)}

        loader = DataLoader(dataset, batch_size=16)

        predictions = []
        confidences = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                max_probs, predicted_ids = torch.max(probs, dim=1)

                predictions.extend(predicted_ids.tolist())
                confidences.extend(max_probs.tolist())

        predicted_labels = self.label_encoder.inverse_transform(predictions)

        return predicted_labels, confidences
