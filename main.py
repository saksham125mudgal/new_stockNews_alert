from fine_tuning import LangChainFinetuner



def main():
# Create prediction pipeline
    finetuner=LangChainFinetuner()
    # pipeline = finetuner.create_prediction_pipeline()
    
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
        print(f"Prediction: {prediction[0]}")
        

if __name__ == "__main__":
    main()