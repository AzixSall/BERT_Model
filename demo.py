import gradio as gr
import torch
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from transformers import BertModel

MAX_LEN = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


def prepare_model_and_data():
    try:
        data = pd.read_csv('movie_plots.csv')
        print("Loaded data")
    except:
        print("Error loading data")

    plots = data['Plot'].values
    origin_categories = sorted(data['Origin/Ethnicity'].unique())
    origin_to_id = {origin: idx for idx, origin in enumerate(origin_categories)}
    origins = [origin_to_id[origin] for origin in data['Origin/Ethnicity']]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model = BertMoviePlotClassifier(n_classes=len(origin_categories))
    
    try:
        model.load_state_dict(torch.load('bert_movie_plot_classifier.pt', map_location=DEVICE))
        print("Loaded pre-trained model")
    except:
        print("Model loading error")
    
    model = model.to(DEVICE)
    model.eval()
    
    return model, tokenizer, origin_categories, data

def predict_origin(model, tokenizer, text, origin_categories):
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
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    token_type_ids = encoding['token_type_ids'].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences = probs.cpu().numpy()[0]
        prediction_idx = torch.argmax(probs, dim=1).item()
    
    results = {origin: float(conf) for origin, conf in zip(origin_categories, confidences)}
    
    plt.figure(figsize=(10, 6))
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    categories = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]
    
    chart = sns.barplot(x=scores, y=categories, palette="viridis")
    plt.title("Prediction Confidence by Origin/Ethnicity")
    plt.xlabel("Confidence Score")
    plt.ylabel("Origin/Ethnicity")
    plt.tight_layout()
    
    chart_path = "confidence_chart.png"
    plt.savefig(chart_path)
    plt.close()
    
    predicted_origin = origin_categories[prediction_idx]
    confidence = confidences[prediction_idx]
    
    return predicted_origin, confidence, chart_path, results

model, tokenizer, origin_categories, sample_data = prepare_model_and_data()

example_plots = sample_data[['Title', 'Plot']].values.tolist()

def predict_plot_origin(plot_text, example_idx=None):
    if example_idx is not None and example_idx >= 0:
        title, plot_text = example_plots[example_idx]
    
    if not plot_text.strip():
        return "Please enter a plot description", 0.0, None, "No prediction made"

    predicted_origin, confidence, chart_path, all_results = predict_origin(
        model, tokenizer, plot_text, origin_categories
    )
    
    prediction_message = f"Predicted Origin: {predicted_origin}"
    confidence_message = f"Confidence: {confidence:.2%}"
    
    all_results_str = "\n".join([f"{origin}: {conf:.2%}" for origin, conf in 
                                sorted(all_results.items(), key=lambda x: x[1], reverse=True)])
    
    return prediction_message, confidence_message, chart_path, all_results_str


with gr.Blocks(title="Movie Plot Origin Classifier") as demo:
    gr.Markdown("# Movie Plot Origin Classifier")
    gr.Markdown("Enter a movie plot description and the model will predict its origin/ethnicity.")
    
    with gr.Row():
        with gr.Column():
            plot_input = gr.Textbox(
                label="Enter movie plot description", 
                placeholder="Type a movie plot here...",
                lines=5
            )
            
            # example_dropdown = gr.Dropdown(
            #     [f"{title}: {plot[:50]}..." for title, plot in example_plots],
            #     label="Or select an example"
            # )
            
            predict_button = gr.Button("Predict Origin")
            
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction")
            confidence_output = gr.Textbox(label="Confidence")
            chart_output = gr.Image(label="Confidence Chart")
            details_output = gr.Textbox(label="Detailed Results", lines=10)
    
    predict_button.click(
        fn=lambda text: predict_plot_origin(text, -1), inputs=plot_input,
        outputs=[prediction_output, confidence_output, chart_output, details_output]
    )
    
    # example_dropdown.change(
    #     fn=lambda idx: predict_plot_origin("", idx),
    #     inputs=example_dropdown,
    #     outputs=[prediction_output, confidence_output, chart_output, details_output]
    # )

if __name__ == "__main__":
    demo.launch()