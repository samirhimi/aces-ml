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
                    # Run unit tests and generate coverage report
                    pytest -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov-fail-under=80 --cov=docker-images/

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



        stage('Deploy to Kubernetes') {
            steps {
                script {
                    // Update Helm chart values with new image tags
                    sh """
                        helm upgrade --install ${APP_NAME} ./aces-ml \
                            --namespace aces-ml-dev \
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