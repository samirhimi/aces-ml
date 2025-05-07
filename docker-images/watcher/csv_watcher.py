#!/usr/bin/env python3

import time
import os
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from datetime import datetime
import argparse
import urllib3
import warnings

# Suppress only the specific InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CSVUploadHandler(FileSystemEventHandler):
    def __init__(self, api_url):
        self.api_url = api_url
        self.api_token = os.getenv('API_TOKEN')
        if not self.api_token:
            raise ValueError("API_TOKEN environment variable is not set")
        self.processed_files = set()
        self.chunk_size = 1000  # Process 1000 rows at a time

    def upload_file(self, file_path):
        if not file_path.endswith('.csv'):
            return
            
        try:
            # Skip if we've already processed this file
            if file_path in self.processed_files:
                return
                
            print(f"🔄 New CSV file detected: {file_path}")
            
            # Wait a brief moment to ensure the file is fully written
            time.sleep(1)
            
            headers = {
                'Authorization': f'Bearer {self.api_token}'
            }
            
            print(f"📤 Processing {file_path} in chunks")
            
            # Read and process the CSV file in chunks
            chunk_number = 0
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk_number += 1
                chunk_file = f"{file_path}.chunk{chunk_number}"
                
                try:
                    # Save the chunk to a temporary file
                    chunk.to_csv(chunk_file, index=False)
                    print(f"📤 Uploading chunk {chunk_number}")
                    
                    with open(chunk_file, 'rb') as f:
                        # Create a session with custom SSL verification settings
                        with requests.Session() as session:
                            # Configure session with proper SSL settings
                            session.verify = False  # Required for self-signed certs
                            adapter = requests.adapters.HTTPAdapter(
                                pool_connections=1,
                                pool_maxsize=1,
                                max_retries=3,
                                pool_block=True
                            )
                            session.mount('https://', adapter)
                            
                            response = session.post(
                                self.api_url,
                                headers=headers,
                                files={'file': (os.path.basename(file_path), f, 'text/csv')},
                                timeout=600  # 10 minutes timeout per chunk
                            )
                        
                    if response.status_code == 200:
                        print(f"✅ Chunk {chunk_number} uploaded successfully: {response.json()}")
                    else:
                        print(f"❌ Chunk {chunk_number} upload failed: {response.status_code} - {response.text}")
                        return
                        
                finally:
                    # Clean up the temporary chunk file
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
            
            print(f"✅ Complete file {file_path} processed and uploaded successfully")
            self.processed_files.add(file_path)
                    
        except Exception as e:
            print(f"❌ Error processing file {file_path}: {str(e)}")

    def on_created(self, event):
        if not event.is_directory:
            self.upload_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.upload_file(event.src_path)

def main():
    parser = argparse.ArgumentParser(description='Watch directory for CSV files and upload to ML API')
    parser.add_argument('--path', required=True, help='Directory path to watch')
    parser.add_argument('--url', required=True, help='API endpoint URL')
    
    args = parser.parse_args()
    
    print(f"🔍 Starting CSV file watcher on directory: {args.path}")
    print(f"🎯 API endpoint: {args.url}")
    
    try:
        event_handler = CSVUploadHandler(args.url)
        observer = Observer()
        observer.schedule(event_handler, args.path, recursive=False)
        observer.start()
        
        while True:
            time.sleep(1)
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        return 1
    except KeyboardInterrupt:
        observer.stop()
        print("\n⏹️ Stopping file watcher...")
        observer.join()
    
    return 0

if __name__ == "__main__":
    exit(main())