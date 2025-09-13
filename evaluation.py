from data_manager import load_prepared_data
from transformers import pipeline
from sklearn.metrics import accuracy_score

# Load processed data
_, val_dataset, label_mapping = load_prepared_data()
texts = val_dataset['text']
true_labels =val_dataset['label']

reverse_label_mapping = {v: k for k, v in label_mapping.items()}
true_label_strings = [reverse_label_mapping[label] for label in true_labels]

# Predict
classifier = pipeline("text-classification", model="./model/finbert-finetuned", top_k=1)
predictions = classifier(texts)
pred_labels = [pred[0]['label'] for pred in predictions]

# Calculate accuracy
accuracy = accuracy_score(true_label_strings, pred_labels)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")