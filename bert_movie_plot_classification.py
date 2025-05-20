import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class MoviePlotDataset(Dataset):
    def __init__(self, plots, origins, tokenizer, max_len):
        self.plots = plots
        self.origins = origins
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.plots)

    def __getitem__(self, idx):
        plot = str(self.plots[idx])
        origin = self.origins[idx]

        # Tokenize plot
        encoding = self.tokenizer.encode_plus(
            plot,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'plot_text': plot,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'origin': torch.tensor(origin, dtype=torch.long)
        }

class BertMoviePlotClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertMoviePlotClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output

        output = self.dropout(pooled_output)
        return self.classifier(output)

def load_and_preprocess_data(file_path):
    data = pd.read_csv('wiki_movie_plots_deduped.csv')

    plots = data['Plot'].values
    origin_categories = data['Origin/Ethnicity'].unique()
    origin_to_id = {origin: idx for idx, origin in enumerate(origin_categories)}
    origins = [origin_to_id[origin] for origin in data['Origin/Ethnicity']]

    return plots, origins, origin_to_id, origin_categories

def train_model():
    plots, origins, origin_to_id, origin_categories = load_and_preprocess_data('wiki_movie_plots_deduped.csv')

    plots_train, plots_val, origins_train, origins_val = train_test_split(
        plots, origins, test_size=0.2, random_state=RANDOM_SEED
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = MoviePlotDataset(
        plots=plots_train,
        origins=origins_train,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = MoviePlotDataset(
        plots=plots_val,
        origins=origins_val,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    model = BertMoviePlotClassifier(n_classes=len(origin_categories))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            origins = batch['origin'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = loss_fn(outputs, origins)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                origins = batch['origin'].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                loss = loss_fn(outputs, origins)
                val_losses.append(loss.item())

                _, preds = torch.max(outputs, dim=1)
                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(origins.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train Loss: {np.mean(train_losses):.4f}')
        print(f'Val Loss: {np.mean(val_losses):.4f}')
        print(f'Val Accuracy: {accuracy:.4f}')
        #print(classification_report(all_labels, all_predictions, target_names=origin_categories))
        print('-' * 50)

    torch.save(model.state_dict(), 'bert_movie_plot_classifier.pt')

    return model, tokenizer, origin_to_id, origin_categories

def predict_origin(model, tokenizer, text, origin_categories, device=DEVICE):

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        _, prediction = torch.max(outputs, dim=1)

    return origin_categories[prediction.item()]

def main():
    model, tokenizer, origin_to_id, origin_categories = train_model()

    sample_plot = "A family moves to a new house in the suburbs and discovers something strange in their basement."
    predicted_origin = predict_origin(model, tokenizer, sample_plot, origin_categories)
    print(f"Predicted Origin for Sample Plot: {predicted_origin}")

if __name__ == "__main__":
    main()