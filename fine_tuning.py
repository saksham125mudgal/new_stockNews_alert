from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
from data_manager import load_prepared_data  # Import your data preparation

print("✓ All imports successful - No TensorFlow!")

class LangChainFinetuner:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load model using LangChain's HuggingFace pipeline"""
        print("Loading model with LangChain...")
        
        # Load tokenizer and model directly
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=3  # negative, neutral, positive
        )
        self.model.to(self.device)
        
        print("✓ Model loaded successfully with LangChain")
    
    def prepare_training_data(self):
        """Load and prepare training data using your existing module"""
        print("Loading prepared data...")
        train_dataset, val_dataset, label_mapping = load_prepared_data()
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Label mapping: {label_mapping}")
        
        return train_dataset, val_dataset, label_mapping
    
    def tokenize_datasets(self, train_dataset, val_dataset):
        """Tokenize the datasets for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=True, 
                max_length=256
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        
        return tokenized_train, tokenized_val
    
    def fine_tune(self):
        """Fine-tune using Hugging Face Trainer"""
        print("Starting fine-tuning process...")
        
        # Load prepared data
        train_dataset, val_dataset, label_mapping = self.prepare_training_data()
        
        # Tokenize datasets
        tokenized_train, tokenized_val = self.tokenize_datasets(train_dataset, val_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./model/finbert-finetuned",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=100,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained("./model/finbert-finetuned")
        
        # Save label mapping
        with open("./model/finbert-finetuned/label_mapping.json", "w") as f:
            json.dump(label_mapping, f)
        
        print("✓ Fine-tuning completed successfully!")
        print("Model saved to './model/finbert-finetuned/'")
        
        return trainer
    
    def create_prediction_pipeline(self):
        """Create LangChain pipeline for predictions after fine-tuning"""
        print("Creating prediction pipeline...")
        
        # Load the fine-tuned model
        self.model = AutoModelForSequenceClassification.from_pretrained("./model/finbert-finetuned")
        self.tokenizer = AutoTokenizer.from_pretrained("./model/finbert-finetuned")
        
        # Create prediction pipeline directly with transformers
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            
        )
        
        
        print("✓ Prediction pipeline created successfully")
        return self.pipeline
    
    def predict(self, text):
        """Make predictions using the fine-tuned model"""
        if not hasattr(self, 'pipeline'):
            self.create_prediction_pipeline()
        
        result = self.pipeline(text)
        return result

# Usage example
def main():
    # Initialize
    print("=== LANGCHAIN FINE-TUNING ===")
    finetuner = LangChainFinetuner()
    
    # Load base model
    finetuner.load_model()
    
    # Fine-tune on your data
    trainer = finetuner.fine_tune()
    
    # Create prediction pipeline
    llm = finetuner.create_prediction_pipeline()
    
    # Test predictions
    test_texts = [
        "Company reports strong quarterly earnings growth",
        "Market downturn affects stock prices negatively",
        "Neutral market conditions with stable performance"
    ]
    
    print("\n=== TEST PREDICTIONS ===")
    for text in test_texts:
        prediction = finetuner.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {prediction[0][0]}")
        print("-" * 50)

if __name__ == "__main__":
    main()