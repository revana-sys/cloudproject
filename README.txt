ML API Project
Overview
This project demonstrates deploying a Machine Learning API using Docker and Kubernetes (Minikube).
The API serves predictions from a pre-trained model (model.pkl) via HTTP endpoints.
Features: - Fully containerized Python app - Automatic Kubernetes deployment - Liveness & Readiness probes for pod health - Easy local testing using Minikube


Repository Structure
cloudproject/
├── app.py                  # Flask API entrypoint
├── cloudproject.py         # ML model inference code in vs code 
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├──  ml-api-deployment.yaml  # Kubernetes Deployment (auto-generated)
├──  ml-api-service.yaml     # Kubernetes Service (auto-generated)
├── cloud_project.ipynb     # run in colab
├── model.pkl               # Pre-trained ML model
└── README.md               # Project instructions

Prerequisites
•	Docker (Install Docker Desktop)
•	Minikube (Install Minikube)
•	kubectl (Install kubectl)
•	Python 3.11+ (optional if running locally)
________________________________________
Quick Start Guide
1. Clone the Repository
git clone https://github.com/revana-sys/cloudproject.git
cd cloudproject
2. Start Minikube
minikube start --driver=docker instead of using virtual box 
3-Build Docker image:
docker build -t yourdockerhubusername/ml-model:latest .
4-Run Docker locally:
docker run -p 5000:5000 yourdockerhubusername/ml-model:latest
5-Once it works, push to Docker Hub:
docker login
docker push yourdockerhubusername/ml-model:latest
6-Deploy using kubectl (automatic YAML generation):
kubectl apply -f ml-api-deployment.yaml --image=yourdockerhubusername/ml-model:latest
7-Expose as service:
kubectl apply -f ml-api-service.yaml --type=NodePort --port=5000
8-Check pods and services:
kubectl get pods-w
kubectl get svc
Access the API
minikube service ml-api-deployment --url
9-Add Health Checks:
kubectl set probe deployment/ml-model --liveness --get-url=http://:5000/healthz --initial-delay-seconds=5 --period-seconds=10
Notes
•	imagePullPolicy: Never is set so Kubernetes uses the local Docker image in        Minikube.
•	Deployment and Service YAML files are auto-generated using kubectl commands.
•	The app is fully self-contained and can be tested locally without external dependencies.

10-Readiness probe (routes traffic only to ready pods):
kubectl set probe deployment/ml-model --readiness --get-url=http://:5000/healthz --initial-delay-seconds=5 --period-seconds=10
Configure Horizontal Pod Autoscaler (HPA):
minikube addons enable metrics-server
Create HPA:
kubectl autoscale deployment ml-model --cpu-percent=50 --min=1 --max=5
Check HPA:
kubectl get hpa

Troubleshooting
•	Pod not ready: Check logs:
kubectl logs <pod-name>
•	Image pull error: Make sure Minikube is using the correct Docker environment:
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
•	Service not reachable: Run:
minikube service ml-api-deployment --url
