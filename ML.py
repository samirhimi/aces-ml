#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X_train, X_test, y_train, y_test):
        print("\n🔹 Training Random Forest model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🔹 Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("🔹 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self.save_model()
        
    def save_model(self, output_dir="models"):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "random_forest_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"✔️ Model saved to {model_path}")

class DatasetHandler(FileSystemEventHandler):
    def __init__(self):
        self.trainer = ModelTrainer()
        self.last_modified = 0
        
    def process_dataset(self, dataset_path):
        # Add a small delay to ensure the file is completely written
        time.sleep(1)
        
        try:
            print(f"📌 Processing dataset: {dataset_path}")
            # Read the dataset
            df = pd.read_csv(dataset_path, low_memory=False)
            
            # Remove rows with NaN values in the target column
            df = df.dropna(subset=['Abnormality class'])
            
            print("📌 Data shape:", df.shape)
            print("📌 Class distribution:")
            print(df['Abnormality class'].value_counts())
            
            # Drop non-numeric columns and unnecessary columns
            drop_columns = ['Unnamed: 0', 'timestamp', 'Microservice', 'Experiment']
            # Find all columns with '_deployed_at' suffix
            deployed_columns = [col for col in df.columns if col.endswith('_deployed_at')]
            drop_columns.extend(deployed_columns)
            
            # Prepare features and target
            X = df.drop(drop_columns + ['Abnormality class'], axis=1)
            y = df['Abnormality class']
            
            # Convert all columns to numeric, replacing non-numeric values with NaN
            X = X.apply(pd.to_numeric, errors='coerce')
            
            # Fill NaN values with 0
            X = X.fillna(0)
            
            print("📌 Features shape after preprocessing:", X.shape)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"✔️ Training set size: {X_train.shape}")
            print(f"✔️ Test set size: {X_test.shape}")
            
            # Train the model
            self.trainer.train(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"❌ Error processing dataset: {str(e)}")
    
    def on_modified(self, event):
        if event.src_path.endswith('.csv'):
            current_time = time.time()
            # Prevent multiple processing of the same event
            if current_time - self.last_modified > 1:
                print(f"\n🔄 Dataset modified: {event.src_path}")
                self.process_dataset(event.src_path)
                self.last_modified = current_time

def main():
    # Create the handler and observer
    handler = DatasetHandler()
    observer = Observer()
    observer.schedule(handler, path=".", recursive=False)
    observer.start()
    
    print("👀 Monitoring for dataset changes...")
    
    # Initial processing if dataset exists
    if os.path.exists('final_dataset.csv'):
        handler.process_dataset('final_dataset.csv')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n⚡ Stopping model training service...")
    observer.join()

if __name__ == "__main__":
    main()
