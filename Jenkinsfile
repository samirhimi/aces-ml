pipeline {

    agent {
        label 'docker'
    }

    environment {
        DOCKER_REGISTRY = 'sami4rhimi'  
        APP_NAME = 'aces-ml'
        DOCKER_IMAGE = "${DOCKER_REGISTRY}/${APP_NAME}"
        DOCKER_CREDS = credentials('docker-credentials')
        KUBECONFIG = credentials('kubeconfig')
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'dev', url: 'https://github.com/samirhimi/aces-ml.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r docker-images/requirements.txt
                    pip install pytest pytest-cov safety bandit
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest --cov=docker-images/. --cov-report=xml
                '''
            }
        }

        stage('Security Scan') {
            parallel {
                stage('Dependencies Check') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            safety check
                        '''
                    }
                }
                stage('Static Code Analysis') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            bandit -r . -f json -o bandit-report.json
                        '''
                    }
                }
            }
        }

        stage('Build Docker Images') {
            steps {
                script {
                    // Build main ML application image
                    docker.build("${DOCKER_IMAGE}-app:${BUILD_NUMBER}", "-f docker-images/Dockerfile ./docker-images")
                }
            }
        }

        stage('scan Docker Images') {
            steps {
                script {
                    // Scan the Docker image for vulnerabilities
                    sh """
                        docker run --rm \
                            -v /var/run/docker.sock:/var/run/docker.sock \
                            aquasec/trivy:latest image \
                            --exit-code 1 \
                            --severity HIGH,CRITICAL \
                            --no-progress \
                            ${DOCKER_IMAGE}-app:${BUILD_NUMBER}
                    """
                }
            }
        }

        stage('Push Docker Images') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-credentials') {
                        docker.image("${DOCKER_IMAGE}-app:${BUILD_NUMBER}").push()
                        docker.image("${DOCKER_IMAGE}-app:${BUILD_NUMBER}").push('latest')
                    }
                }
            }
        }

        stage('Validate Dataset') {
            steps {
                script {
                    // Check for dataset file and content
                    def validationResult = sh(script: '''
                        # Check if dataset exists
                        if [ ! -f "docker-images/final_dataset.csv" ]; then
                            echo "Dataset file not found"
                            exit 1
                        fi

                        # Check if dataset has content beyond header
                        lines=$(wc -l < docker-images/final_dataset.csv)
                        if [ "$lines" -le 1 ]; then
                            echo "Dataset is empty or contains only headers"
                            exit 1
                        fi

                        # Verify CSV structure
                        python3 -c "
import pandas as pd
try:
    df = pd.read_csv('docker-images/final_dataset.csv')
    if df.empty:
        raise Exception('Dataset is empty')
    print(f'Dataset validated successfully with {len(df)} records')
except Exception as e:
    print(f'Dataset validation failed: {str(e)}')
    exit(1)
"
                    ''', returnStatus: true)

                    if (validationResult != 0) {
                        error "Dataset validation failed: No data was collected"
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    // Update Helm chart values with new image tags and monitoring
                    sh """
                        helm upgrade --install ${APP_NAME} ./aces-ml \
                            --namespace aces-ml-dev \
                            --create-namespace \
                            --set image.repository=${DOCKER_IMAGE}-app \
                            --set image.tag=${BUILD_NUMBER} \
                            --set env[0].name=LOG_LEVEL \
                            --set env[0].value=INFO \
                            --set env[1].name=MIN_SAMPLES_FOR_TRAINING \
                            --set env[1].value=1 \
                            --wait --timeout 10m
                    """

                    // Wait for deployment and verify data collection
                    sh """
                        # Wait for pods to be ready
                        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=${APP_NAME} -n aces-ml-dev --timeout=300s

                        # Monitor pod logs for data collection status
                        POD=\$(kubectl get pod -n aces-ml-dev -l app.kubernetes.io/name=${APP_NAME} -o jsonpath='{.items[0].metadata.name}')
                        
                        # Check for no-data-collected warning
                        echo "Monitoring application logs for data collection status..."
                        TIMEOUT=300
                        START_TIME=\$(date +%s)
                        
                        while true; do
                            CURRENT_TIME=\$(date +%s)
                            if [ \$((CURRENT_TIME - START_TIME)) -gt \$TIMEOUT ]; then
                                echo "Timeout waiting for data collection status"
                                exit 1
                            fi
                            
                            if kubectl logs -n aces-ml-dev \$POD --tail=50 | grep -q "No data was collected"; then
                                echo "WARNING: Application reported no data collected"
                                kubectl logs -n aces-ml-dev \$POD
                                exit 1
                            fi
                            
                            if kubectl logs -n aces-ml-dev \$POD | grep -q "Data collection completed successfully"; then
                                echo "Data collection verified successfully"
                                break
                            fi
                            
                            sleep 10
                        done
                    """
                }
            }
        }
    }

    post {
        always {
            // Clean up workspace
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed! Please check the logs for details.'
        }
    }
}