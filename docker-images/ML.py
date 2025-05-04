#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import time
import shutil
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from functools import wraps
import threading
import socket
import psutil
import queue
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# Create a queue for new metrics
metrics_queue = queue.Queue()

# Create an event to signal training thread to stop
stop_training = threading.Event()

def require_api_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_token = os.getenv('API_TOKEN')
        if not api_token:
            return jsonify({"error": "API token not configured"}), 500
            
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "Authorization header is missing"}), 401
            
        try:
            # Extract token from "Bearer <token>"
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                return jsonify({"error": "Invalid authorization type. Use 'Bearer <token>'"}), 401
                
            if token != api_token:
                return jsonify({"error": "Invalid API token"}), 401
                
        except ValueError:
            return jsonify({"error": "Invalid authorization format. Use 'Bearer <token>'"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_lock = threading.Lock()
        self.dataset_lock = threading.Lock()
        self.training = False
        self.dataset_path = os.getenv('DATASET_PATH', 'final_dataset.csv')
        self.backup_dir = 'csv_backups'
        self.min_samples_for_training = int(os.getenv('MIN_SAMPLES_FOR_TRAINING', '1'))
        self.last_training_time = None
        os.makedirs(self.backup_dir, exist_ok=True)
        self.load_model()
        # Start CSV cleanup thread
        self.cleanup_thread = threading.Thread(target=self.cleanup_old_files, daemon=True)
        self.cleanup_thread.start()

    def load_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "random_forest_model.joblib")
        if os.path.exists(model_path):
            with self.model_lock:
                self.model = joblib.load(model_path)
                print(f"✔️ Model loaded from {model_path}")
        else:
            print(f"⚠️ No existing model found at {model_path}, will train a new one")

    def backup_current_dataset(self):
        """Backup the current dataset with timestamp"""
        if os.path.exists(self.dataset_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f'final_dataset_{timestamp}.csv'
            backup_path = os.path.join(self.backup_dir, backup_filename)
            shutil.copy2(self.dataset_path, backup_path)
            print(f"✔️ Dataset backed up to {backup_path}")

    def cleanup_old_files(self):
        """Cleanup files older than 1 day in the backup directory"""
        while not stop_training.is_set():
            try:
                current_time = datetime.now()
                for filename in os.listdir(self.backup_dir):
                    file_path = os.path.join(self.backup_dir, filename)
                    file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if current_time - file_modified_time > timedelta(days=1):
                        os.remove(file_path)
                        print(f"🗑️ Removed old backup file: {filename}")
            except Exception as e:
                print(f"Error during file cleanup: {str(e)}")
            # Sleep for 1 hour before next cleanup check
            time.sleep(3600)

    def append_metrics(self, metrics_data):
        """Append new metrics to the dataset file"""
        with self.dataset_lock:
            # Backup current dataset before appending new data
            self.backup_current_dataset()
            df = pd.DataFrame([metrics_data])
            df.to_csv(self.dataset_path, mode='a', header=False, index=False)
            print(f"✔️ Appended new metrics to {self.dataset_path}")

    def should_train(self):
        """Check if we should retrain the model"""
        try:
            if not os.path.exists(self.dataset_path):
                return False
            df = pd.read_csv(self.dataset_path)
            
            # Always train if we have at least min_samples_for_training records
            if len(df) >= self.min_samples_for_training:
                # If this is the first training or we have new data
                if not self.last_training_time or os.path.getmtime(self.dataset_path) > self.last_training_time:
                    return True
            return False
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
                success = self.save_model()
                if success:
                    self.last_training_time = time.time()
                    # Broadcast model update through a socket or event system
                    print("✨ New model is now available")
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
                    return True
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
            return False

def process_dataset(dataset_path, trainer):
    time.sleep(1)
    try:
        print(f"📌 Processing dataset: {dataset_path}")
        
        # Ensure model directory exists
        model_dir = os.path.dirname(trainer.ML_MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        
        df = pd.read_csv(dataset_path, low_memory=False)
        
        # Check if 'Abnormality class' column exists
        if 'Abnormality class' not in df.columns:
            print(f"❌ Error: Required column 'Abnormality class' not found in dataset")
            print(f"Available columns: {df.columns.tolist()}")
            return
            
        # Ensure consistent number of columns
        expected_cols = 265
        if len(df.columns) > expected_cols:
            print(f"⚠️ Found {len(df.columns)} columns, truncating to {expected_cols}")
            df = df.iloc[:, :expected_cols]
        elif len(df.columns) < expected_cols:
            print(f"⚠️ Found {len(df.columns)} columns, padding to {expected_cols}")
            for i in range(len(df.columns), expected_cols):
                df[f'col_{i}'] = 0
            
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
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Dataset path: {dataset_path}")
        print(f"📁 Dataset exists: {os.path.exists(dataset_path)}")
        if os.path.exists(dataset_path):
            print(f"📁 Dataset permissions: {oct(os.stat(dataset_path).st_mode)[-3:]}")
            print(f"📁 Dataset size: {os.path.getsize(dataset_path)} bytes")

def process_metrics_queue():
    """Background thread to process metrics and retrain model"""
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
    batch_timeout = float(os.getenv('BATCH_TIMEOUT', '1.0'))
    batch = []
    last_process_time = time.time()

    while not stop_training.is_set():
        try:
            # Try to get a metric, wait up to batch_timeout seconds
            try:
                metric = metrics_queue.get(timeout=batch_timeout)
                batch.append(metric)
            except queue.Empty:
                # If we have any items in the batch, process them
                if batch:
                    for item in batch:
                        trainer.append_metrics(item)
                    if trainer.should_train():
                        process_dataset(trainer.dataset_path, trainer)
                    batch = []
                continue

            # If we've reached batch size or batch timeout, process the batch
            current_time = time.time()
            if len(batch) >= batch_size or (current_time - last_process_time) >= batch_timeout:
                for item in batch:
                    trainer.append_metrics(item)
                if trainer.should_train():
                    process_dataset(trainer.dataset_path, trainer)
                batch = []
                last_process_time = current_time

        except Exception as e:
            print(f"Error processing metrics: {str(e)}")
            batch = []  # Clear batch on error

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

@app.route('/model-status', methods=['GET'])
def model_status():
    """Get the current model status including last training time"""
    model_path = os.path.join("models", "random_forest_model.joblib")
    if os.path.exists(model_path):
        file_stats = os.stat(model_path)
        last_modified = file_stats.st_mtime
        return jsonify({
            "status": "available",
            "last_modified": time.ctime(last_modified),
            "last_modified_timestamp": last_modified,
            "size_bytes": file_stats.st_size,
            "last_training_time": trainer.last_training_time,
            "is_training": trainer.training
        })
    else:
        return jsonify({
            "status": "unavailable",
            "is_training": trainer.training
        })

@app.route('/metrics', methods=['POST'])
@require_api_token
def receive_metrics():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
                
            if not file.filename.endswith('.csv'):
                return jsonify({"error": "Only CSV files are supported"}), 400
                
            # Read CSV file
            try:
                # Read CSV file into memory
                stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
                df = pd.read_csv(stream)
                
                # Validate required columns
                required_fields = ['timestamp', 'Abnormality class']
                missing_fields = [field for field in required_fields if field not in df.columns]
                if missing_fields:
                    return jsonify({"error": f"Missing required columns in CSV: {', '.join(missing_fields)}"}), 400
                
                # Process each row
                records_processed = 0
                for _, row in df.iterrows():
                    metrics_queue.put(row.to_dict())
                    records_processed += 1
                
                return jsonify({
                    "status": "success",
                    "message": "CSV file processed successfully",
                    "records_processed": records_processed
                }), 200
                
            except pd.errors.EmptyDataError:
                return jsonify({"error": "The CSV file is empty"}), 400
            except Exception as e:
                return jsonify({"error": f"Error processing CSV file: {str(e)}"}), 400
                
        elif request.is_json:
            metrics = request.json
            if not metrics:
                return jsonify({"error": "No metrics data provided"}), 400
                
            # Validate required fields
            required_fields = ['timestamp', 'Abnormality class']
            missing_fields = [field for field in required_fields if field not in metrics]
            if missing_fields:
                return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
                
            # Add metrics to processing queue
            metrics_queue.put(metrics)
            return jsonify({
                "status": "success",
                "message": "Metrics received and queued for processing",
                "records_processed": 1
            }), 200
        else:
            return jsonify({"error": "Request must be either JSON data or a CSV file"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

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
        required_libraries = ['pandas', 'numpy', 'sklearn', 'joblib', 'flask']
        missing_libraries = []
        
        for lib in required_libraries:
            try:
                __import__(lib)
            except ImportError:
                missing_libraries.append(lib)
        
        if (missing_libraries):
            print(f"❌ Error: Missing required libraries: {', '.join(missing_libraries)}")
            print("Please install them using: pip install " + " ".join(missing_libraries))
            return
            
        global trainer
        trainer = ModelTrainer()
        
        # Start metrics processing thread
        metrics_thread = threading.Thread(target=process_metrics_queue, daemon=True)
        metrics_thread.start()
        
        if os.path.exists(trainer.dataset_path):
            print(f"📋 Found existing dataset at {trainer.dataset_path}, processing...")
            process_dataset(trainer.dataset_path, trainer)
        else:
            print(f"⚠️ Dataset '{trainer.dataset_path}' not found. Creating new dataset.")
            # Create empty dataset with headers
            os.makedirs(os.path.dirname(trainer.dataset_path), exist_ok=True)
            pd.DataFrame(columns=['timestamp', 'Abnormality class']).to_csv(trainer.dataset_path, index=False)
        
        port = find_available_port(8080)
        print(f"🚀 Starting API server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
    finally:
        stop_training.set()  # Signal the training thread to stop

if __name__ == "__main__":
    main()