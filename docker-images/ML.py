#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import time
from flask import Flask, request, jsonify, send_file
from flask_swagger_ui import get_swaggerui_blueprint
import threading
import socket
import psutil
from datetime import datetime
import queue

app = Flask(__name__)

# Create a queue for new metrics
metrics_queue = queue.Queue()

# Create an event to signal training thread to stop
stop_training = threading.Event()

os.makedirs('static', exist_ok=True)
with open('static/swagger.json', 'w') as f:
    f.write('''{
      "swagger": "2.0",
      "info": {
        "title": "ML Model Service API",
        "version": "1.0.0"
      },
      "paths": {
        "/health": {
          "get": {
            "summary": "Health check endpoint"
          }
        },
        "/download-model": {
          "get": {
            "summary": "Download the trained model"
          }
        },
        "/model-info": {
          "get": {
            "summary": "Get information about the model"
          }
        },
        "/metrics": {
          "post": {
            "summary": "Submit new metrics data"
          }
        }
      }
    }''')

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "ML Model Service"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_lock = threading.Lock()
        self.dataset_lock = threading.Lock()
        self.training = False
        self.dataset_path = 'final_dataset.csv'
        self.load_model()

    def load_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "random_forest_model.joblib")
        if os.path.exists(model_path):
            with self.model_lock:
                self.model = joblib.load(model_path)
                print(f"✔️ Model loaded from {model_path}")
        else:
            print(f"⚠️ No existing model found at {model_path}, will train a new one")

    def append_metrics(self, metrics_data):
        """Append new metrics to the dataset file"""
        with self.dataset_lock:
            df = pd.DataFrame([metrics_data])
            df.to_csv(self.dataset_path, mode='a', header=False, index=False)
            print(f"✔️ Appended new metrics to {self.dataset_path}")

    def should_train(self):
        """Check if we should retrain the model"""
        try:
            if not os.path.exists(self.dataset_path):
                return False
            df = pd.read_csv(self.dataset_path)
            # Train if we have at least 100 new records
            return len(df) >= 100
        except Exception as e:
            print(f"Error checking dataset: {str(e)}")
            return False

    def train(self, X_train, X_test, y_train, y_test):
        if self.training:
            print("⚠️ Training already in progress, skipping...")
            return
            
        self.training = True
        try:
            print("\n🔹 Training Random Forest model...")
            with self.model_lock:
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"🔹 Accuracy: {accuracy:.4f}")
                print("🔹 Classification Report:")
                print(classification_report(y_test, y_pred))
                self.save_model()
        finally:
            self.training = False

    def save_model(self, output_dir="models"):
        try:
            print(f"📝 Attempting to save model to directory: {output_dir}")
            abs_output_dir = os.path.abspath(output_dir)
            print(f"📝 Absolute path: {abs_output_dir}")
            
            # Monitor memory usage
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
            print(f"📊 Memory usage before save: {mem_before:.2f} MB")
            
            # Create directory with verbose output
            try:
                os.makedirs(abs_output_dir, exist_ok=True)
                print(f"📁 Created/verified directory: {abs_output_dir}")
                print(f"📁 Directory permissions: {oct(os.stat(abs_output_dir).st_mode)[-3:]}")
            except Exception as dir_error:
                print(f"❌ Error creating directory: {str(dir_error)}")
                raise
            
            model_path = os.path.join(abs_output_dir, "random_forest_model.joblib")
            print(f"📝 Full model path: {model_path}")
            
            print("💾 Starting model dump...")
            try:
                # Use compression to reduce memory usage during save
                temp_path = model_path + '.tmp'
                print(f"💾 Saving to temporary file: {temp_path}")
                
                # Try to save with explicit file opening
                with open(temp_path, 'wb') as f:
                    joblib.dump(self.model, f, compress=3)
                
                mem_after = process.memory_info().rss / 1024 / 1024
                print(f"📊 Memory usage after dump: {mem_after:.2f} MB (Change: {mem_after - mem_before:.2f} MB)")
                
                print("✅ Temporary file saved successfully")
                print(f"📁 Temporary file permissions: {oct(os.stat(temp_path).st_mode)[-3:]}")
                
                os.replace(temp_path, model_path)
                print("✅ Model file moved to final location")
                
                if os.path.exists(model_path):
                    size = os.path.getsize(model_path)
                    print(f"✔️ Model saved successfully to {model_path} (size: {size:,} bytes)")
                    print(f"📁 Final file permissions: {oct(os.stat(model_path).st_mode)[-3:]}")
                else:
                    raise FileNotFoundError(f"Model file not found after moving from temporary location")
            except Exception as save_error:
                print(f"❌ Error during model save operation: {str(save_error)}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        print("🧹 Cleaned up temporary file")
                    except Exception as cleanup_error:
                        print(f"⚠️ Failed to clean up temporary file: {str(cleanup_error)}")
                raise
        except Exception as e:
            print(f"❌ Error in save_model: {str(e)}")
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Directory listing of {output_dir}:")
            try:
                print(os.listdir(output_dir))
            except Exception as dir_error:
                print(f"(unable to list directory: {str(dir_error)})")
            print("📁 Directory listing of current directory:")
            print(os.listdir('.'))

def process_dataset(dataset_path, trainer):
    time.sleep(1)
    try:
        print(f"📌 Processing dataset: {dataset_path}")
        df = pd.read_csv(dataset_path, low_memory=False)
        
        # Check if 'Abnormality class' column exists
        if 'Abnormality class' not in df.columns:
            print(f"❌ Error: Required column 'Abnormality class' not found in dataset")
            print(f"Available columns: {df.columns.tolist()}")
            return
            
        df = df.dropna(subset=['Abnormality class'])
        print("📌 Data shape:", df.shape)
        print("📌 Class distribution:")
        print(df['Abnormality class'].value_counts())
        
        # FIXED: Safer column dropping approach
        # Only drop columns that actually exist in the dataframe
        drop_cols = ['timestamp', 'Microservice', 'Experiment']
        if 'Unnamed: 0' in df.columns:
            drop_cols.append('Unnamed: 0')
            
        # Find deployed columns
        deployed_cols = [col for col in df.columns if col.endswith('_deployed_at')]
        drop_cols.extend(deployed_cols)
        
        # Make sure we don't try to drop columns that don't exist
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        
        # Make sure 'Abnormality class' exists before trying to drop it
        columns_to_drop = existing_drop_cols.copy()
        if 'Abnormality class' in df.columns:
            columns_to_drop.append('Abnormality class')
            
        X = df.drop(columns_to_drop, axis=1)
        y = df['Abnormality class']
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        print("📌 Features shape after preprocessing:", X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✔️ Training set size: {X_train.shape}")
        print(f"✔️ Test set size: {X_test.shape}")
        trainer.train(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")

def process_metrics_queue():
    """Background thread to process metrics and retrain model"""
    while not stop_training.is_set():
        try:
            metrics = metrics_queue.get(timeout=5)  # Wait up to 5 seconds for new metrics
            trainer.append_metrics(metrics)
            
            if trainer.should_train():
                process_dataset(trainer.dataset_path, trainer)
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error processing metrics: {str(e)}")

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/download-model', methods=['GET'])
def download_model():
    model_path = os.path.join("models", "random_forest_model.joblib")
    if os.path.exists(model_path):
        modified_time = time.ctime(os.path.getmtime(model_path))
        try:
            response = send_file(
                model_path,
                mimetype="application/octet-stream",
                as_attachment=True,
                download_name="random_forest_model.joblib"
            )
            response.headers["X-Model-Last-Modified"] = modified_time
            return response
        except Exception as e:
            return jsonify({"error": f"Error sending file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Model file not found"}), 404

@app.route('/model-info', methods=['GET'])
def model_info():
    model_path = os.path.join("models", "random_forest_model.joblib")
    if os.path.exists(model_path):
        file_stats = os.stat(model_path)
        return jsonify({
            "model_path": model_path,
            "last_modified": time.ctime(file_stats.st_mtime),
            "size_bytes": file_stats.st_size
        })
    else:
        return jsonify({"error": "Model file not found"}), 404

@app.route('/metrics', methods=['POST'])
def receive_metrics():
    try:
        metrics = request.json
        if not metrics:
            return jsonify({"error": "No metrics data provided"}), 400
            
        # Validate required fields
        required_fields = ['timestamp', 'Abnormality class']  # Add your required metrics fields
        missing_fields = [field for field in required_fields if field not in metrics]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
            
        # Add metrics to processing queue
        metrics_queue.put(metrics)
        return jsonify({"status": "success", "message": "Metrics received and queued for processing"}), 200
        
    except Exception as e:
        return jsonify({"error": f"Error processing metrics: {str(e)}"}), 500

# FIXED: Add port availability check
def is_port_available(port, host='0.0.0.0'):
    """Check if a port is available on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False

def find_available_port(start_port=8080, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise OSError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

def main():
    try:
        # Verify required libraries are installed
        required_libraries = ['pandas', 'numpy', 'sklearn', 'joblib', 'flask', 'flask_swagger_ui']
        missing_libraries = []
        
        for lib in required_libraries:
            try:
                __import__(lib)
            except ImportError:
                missing_libraries.append(lib)
        
        if missing_libraries:
            print(f"❌ Error: Missing required libraries: {', '.join(missing_libraries)}")
            print("Please install them using: pip install " + " ".join(missing_libraries))
            return
            
        global trainer
        trainer = ModelTrainer()
        
        # Start metrics processing thread
        metrics_thread = threading.Thread(target=process_metrics_queue, daemon=True)
        metrics_thread.start()
        
        if os.path.exists('final_dataset.csv'):
            print("📋 Found existing dataset, processing...")
            process_dataset('final_dataset.csv', trainer)
        else:
            print("⚠️ Dataset 'final_dataset.csv' not found. Creating new dataset.")
            # Create empty dataset with headers
            pd.DataFrame(columns=['timestamp', 'Abnormality class']).to_csv('final_dataset.csv', index=False)
        
        port = find_available_port(8080)
        print(f"🚀 Starting API server on port {port} (with Swagger)...")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
    finally:
        stop_training.set()  # Signal the training thread to stop

if __name__ == "__main__":
    main()