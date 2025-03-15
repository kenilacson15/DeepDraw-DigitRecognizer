import os
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Add src to the path properly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

# Import from src
from gui.digit_recognizer import DigitRecognizer
from model.train import train_and_save_model

def main():
    parser = argparse.ArgumentParser(description='Digit Recognizer Application')
    parser.add_argument('--train', action='store_true', help='Force training a new model')
    parser.add_argument('--model-path', default='models/mnist_model.keras', 
                        help='Path to the model file (relative to project root)')
    parser.add_argument('--analyze-feedback', action='store_true', help='Analyze user feedback data')
    args = parser.parse_args()
    
    # Resolve model path properly
    model_path = os.path.join(BASE_DIR, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    
    if args.analyze_feedback:
        analyze_feedback()
        return
    
    if args.train and os.path.exists(model_path):
        print(f"Removing existing model at {model_path}")
        os.remove(model_path)
    
    if not os.path.exists(model_path):
        print(f"Training a new model and saving to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        train_and_save_model(model_path)
    else:
        print(f"Using existing model at {model_path}")
    
    app = DigitRecognizer(model_path)
    app.run()

def analyze_feedback():
    feedback_file = os.path.join(BASE_DIR, 'data', 'feedback', 'feedback_log.jsonl')
    
    if not os.path.exists(feedback_file):
        print("No feedback data found.")
        return
    
    # Load feedback data efficiently
    feedback_data = [json.loads(line) for line in open(feedback_file, 'r')]
    df = pd.DataFrame(feedback_data)
    
    # Extract prediction info
    df['predicted_digit'] = df['prediction'].apply(lambda x: x['digit'])
    df['confidence'] = df['prediction'].apply(lambda x: x['confidence'])
    
    # Calculate and display accuracy
    accuracy = df['is_correct'].mean() * 100
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    digit_accuracy = df.groupby('predicted_digit')['is_correct'].mean() * 100
    print("\nAccuracy by digit:")
    print(digit_accuracy.to_string())
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(BASE_DIR, 'data', 'feedback')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    plt.figure(figsize=(10, 6))
    digit_accuracy.plot(kind='bar')
    plt.title('Accuracy by Digit')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'accuracy_by_digit.png'))
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column='confidence', by=['predicted_digit', 'is_correct'])
    plt.title('Confidence Distribution by Digit and Correctness')
    plt.suptitle('')
    plt.xlabel('(Digit, Is Correct)')
    plt.ylabel('Confidence (%)')
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    
    print("\nAnalysis complete. Plots saved to data/feedback/")

if __name__ == "__main__":
    # Reduce TensorFlow verbosity
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
