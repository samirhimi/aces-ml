#!/usr/bin/env python3

import pandas as pd
import time
import random
from datetime import datetime, timedelta
from kubernetes import client, config
import os
import subprocess

class MetricsInjector:
    def __init__(self, csv_file='final_dataset.csv'):
        self.csv_file = csv_file
        # Read the last row to continue from there
        self.df = pd.read_csv(csv_file)
        self.last_row = self.df.iloc[-1]
        self.last_id = int(self.df.index[-1])
        self.last_timestamp = datetime.strptime(self.last_row[1], '%Y-%m-%d %H:%M:%S')
        
        # Initialize Kubernetes client
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
        self.k8s_client = client.CoreV1Api()
        
        # Get pod name
        self.namespace = 'aces-ml'
        self.pod_name = self.get_aces_ml_pod()
        print(f"Found pod: {self.pod_name}")

    def get_aces_ml_pod(self):
        """Get the aces-ml pod name"""
        pods = self.k8s_client.list_namespaced_pod(namespace=self.namespace, label_selector='app=aces-ml')
        if not pods.items:
            raise Exception("No aces-ml pod found!")
        return pods.items[0].metadata.name

    def generate_next_metrics(self):
        """Generate next set of metrics based on previous values with some variation"""
        new_row = self.last_row.copy()
        
        # Increment ID and timestamp
        self.last_id += 1
        self.last_timestamp += timedelta(seconds=1)
        new_row[0] = self.last_id
        new_row[1] = self.last_timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # Update numeric metrics with random variations (±5%)
        for i in range(2, len(new_row)-3):  # Exclude ID, timestamp, and last 3 columns
            if isinstance(new_row[i], (int, float)):
                variation = random.uniform(-0.05, 0.05)
                new_row[i] = float(new_row[i]) * (1 + variation)

        # Update cart service packet loss (last column, maintaining 0 or 1)
        new_row[-1] = random.choices([0, 1], weights=[0.95, 0.05])[0]
        
        return new_row

    def inject_metrics(self):
        """Inject new metrics into the pod's CSV file"""
        new_row = self.generate_next_metrics()
        row_str = ','.join(str(x) for x in new_row)
        
        # Use kubectl exec to append the row
        cmd = [
            'kubectl', 'exec', '-n', self.namespace, self.pod_name, '--',
            'bash', '-c', f'echo "{row_str}" >> /app/data/final_dataset.csv'
        ]
        subprocess.run(cmd, check=True)
        
        # Update last row
        self.last_row = new_row

    def run(self):
        """Run the injector continuously"""
        print(f"Starting metrics injection to pod {self.pod_name}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.inject_metrics()
                time.sleep(1)  # Wait for 1 second
        except KeyboardInterrupt:
            print("\nStopping metrics injection")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

if __name__ == "__main__":
    injector = MetricsInjector()
    injector.run()