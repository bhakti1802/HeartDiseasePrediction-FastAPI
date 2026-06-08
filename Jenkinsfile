pipeline {

    agent any

    stages {

        stage('Clone Source') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '''
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Verify Imports') {
            steps {
                bat '''
                python -c "import fastapi"
                python -c "import sklearn"
                python -c "import mysql.connector"
                '''
            }
        }

        stage('Start FastAPI') {
            steps {
                bat '''
                taskkill /F /IM python.exe || exit 0
                start "" uvicorn main:app --host 0.0.0.0 --port 8000
                '''
            }
        }
    }

    post {
        success {
            echo 'FastAPI Deployment Successful'
        }

        failure {
            echo 'Deployment Failed'
        }
    }
}