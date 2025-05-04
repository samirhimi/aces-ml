#!/usr/bin/env python3

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from datetime import datetime
import argparse

class CSVUploadHandler(FileSystemEventHandler):
    def __init__(self, api_url):
        self.api_url = api_url
        self.api_token = os.getenv('API_TOKEN')
        if not self.api_token:
            raise ValueError("API_TOKEN environment variable is not set")
        self.processed_files = set()

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
            
            with open(file_path, 'rb') as f:
                headers = {
                    'Authorization': f'Bearer {self.api_token}'
                }
                files = {
                    'file': (os.path.basename(file_path), f, 'text/csv')
                }
                
                print(f"📤 Uploading {file_path} to {self.api_url}")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    verify=False  # Since we're using self-signed certs
                )
                
                if response.status_code == 200:
                    print(f"✅ Upload successful: {response.json()}")
                    self.processed_files.add(file_path)
                else:
                    print(f"❌ Upload failed: {response.status_code} - {response.text}")
                    
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