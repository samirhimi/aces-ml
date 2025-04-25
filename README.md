# AKS Anomaly Detection ML Service

This project implements a real-time machine learning service for detecting anomalies in Azure Kubernetes Service (AKS) metrics. It continuously monitors a dataset of AKS metrics and trains a Random Forest model to classify packet loss anomalies.

## Features

- Real-time dataset monitoring
- Automated model training on dataset updates
- Packet loss anomaly classification
- Model persistence and versioning
- Docker containerization for easy deployment

## Requirements

- Python 3.9+
- Docker
- Required Python packages (see requirements.txt)

## Project Structure

```
.
├── Dockerfile           # Docker configuration
├── ML.py               # Main ML training script
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── models/            # Directory for saved models
    └── random_forest_model.joblib
```

## Quick Start

1. Build the Docker image:
```bash
docker build -t ml-realtime .
```

2. Run the container:
```bash
docker run -it --rm -v $(pwd):/app -v $(pwd)/models:/app/models ml-realtime python ML.py
```

## Dataset Format

The service expects a CSV file named `final_dataset.csv` with the following columns:
- Timestamp
- Various AKS metrics (CPU, Memory, Network metrics)
- Abnormality class (target variable)

## Model Details

- Algorithm: Random Forest Classifier
- Features: AKS performance metrics
- Target: Packet Loss anomaly detection
- Model storage: Saved as .joblib file in models/ directory

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

MIT License