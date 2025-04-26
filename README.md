# AKS Anomaly Detection ML Service

This project implements a real-time machine learning service for detecting anomalies in Azure Kubernetes Service (AKS) metrics. It continuously monitors a dataset of AKS metrics and trains a Random Forest model to classify packet loss anomalies.

## Features

- Real-time dataset monitoring
- Automated model training on dataset updates
- Packet loss anomaly classification
- Model persistence and versioning
- Docker containerization for easy deployment
- Kubernetes deployment via Helm chart

## Requirements

- Python 3.9+
- Docker
- Kubernetes cluster
- Helm 3.x
- Required Python packages (see requirements.txt)

## Project Structure

```
.
├── Dockerfile           # Docker configuration
├── ML.py               # Main ML training script
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── final_dataset.csv   # Training dataset
├── aces-ml-chart/     # Helm chart for Kubernetes deployment
│   ├── Chart.yaml     # Chart metadata
│   ├── values.yaml    # Default configuration values
│   └── templates/     # Kubernetes manifest templates
└── models/            # Directory for saved models
    └── random_forest_model.joblib
```

## Quick Start

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ml-realtime .
```

2. Run the container:
```bash
docker run -it --rm -v $(pwd):/app -v $(pwd)/models:/app/models ml-realtime python ML.py
```

### Kubernetes Deployment

1. Add the Helm repository (if hosted):
```bash
helm repo add aces-ml https://your-helm-repo-url
helm repo update
```

2. Install the chart:
```bash
helm install aces-ml ./aces-ml-chart
```

3. Configure the deployment (optional):
```bash
helm install aces-ml ./aces-ml-chart --values custom-values.yaml
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

## Helm Chart Configuration

The Helm chart provides the following customization options:

- Replica count and autoscaling configuration
- Resource limits and requests
- Ingress configuration
- Service type and port configuration
- Environment variables via ConfigMap

For detailed configuration options, see the [values.yaml](aces-ml-chart/values.yaml) file.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

MIT License