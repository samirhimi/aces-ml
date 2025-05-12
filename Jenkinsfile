pipeline {
    agent linux

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
                checkout scm
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
                    pytest --cov=. --cov-report=xml
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

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    // Update Helm chart values with new image tags
                    sh """
                        helm upgrade --install ${APP_NAME} ./aces-ml \
                            --namespace aces-ml \
                            --create-namespace \
                            --set image.repository=${DOCKER_IMAGE}-app \
                            --set image.tag=${BUILD_NUMBER} \
                            --wait --timeout 10m
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