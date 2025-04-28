#!/usr/bin/env python3

import pandas as pd
import time
import random
from datetime import datetime, timedelta
import os

class MetricsInjector:
    def __init__(self, csv_file='final_dataset.csv'):
        self.csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_file)
        print(f"Using dataset file: {self.csv_file}")
        
        # Read the last row to continue from there
        self.df = pd.read_csv(self.csv_file, low_memory=False)
        if len(self.df) == 0:
            raise ValueError("Dataset is empty")
            
        self.last_row = self.df.iloc[-1]
        self.last_id = int(self.df.index[-1])
        # Get timestamp from the second column (index 1)
        self.last_timestamp = datetime.strptime(self.last_row.iloc[1], '%Y-%m-%d %H:%M:%S')

    def generate_next_metrics(self):
        """Generate next set of metrics based on previous values with some variation"""
        new_row = self.last_row.copy()
        
        # Increment ID and timestamp
        self.last_id += 1
        self.last_timestamp += timedelta(seconds=1)
        new_row.iloc[0] = self.last_id
        new_row.iloc[1] = self.last_timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # Update numeric metrics with random variations (±5%)
        for i in range(2, len(new_row)-3):  # Exclude ID, timestamp, and last 3 columns
            if isinstance(new_row.iloc[i], (int, float)):
                variation = random.uniform(-0.05, 0.05)
                new_row.iloc[i] = float(new_row.iloc[i]) * (1 + variation)

        # Update cart service packet loss (last column, maintaining 0 or 1)
        new_row.iloc[-1] = random.choices([0, 1], weights=[0.95, 0.05])[0]
        
        return new_row

    def inject_metrics(self):
        """Inject new metrics into the CSV file"""
        new_row = self.generate_next_metrics()
        row_str = ','.join(str(x) for x in new_row.values)
        
        try:
            with open(self.csv_file, 'a') as f:
                f.write(f"\n{row_str}")
            print(f"Successfully injected metrics at {self.last_timestamp}")
            # Update last row
            self.last_row = new_row
        except Exception as e:
            print(f"Failed to inject metrics: {str(e)}")
            raise

    def run(self):
        """Run the injector continuously"""
        print("Starting metrics injection")
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