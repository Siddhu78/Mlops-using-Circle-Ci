## Iris Species Prediction with CircleCI
Project Overview
This project demonstrates an end-to-end MLOps workflow using the popular Iris dataset to build a machine learning model that predicts the species of Iris flowers. 
The model is trained using Python’s scikit-learn library, and the entire pipeline is integrated with CircleCI for continuous integration and continuous deployment (CI/CD).

## About CircleCI
CircleCI is a cloud-based CI/CD platform that automates software builds, tests, and deployments. It enables development teams to automate the entire lifecycle of their projects, ensuring that code changes are reliably tested and deployed with minimal manual intervention. In this project, CircleCI is used to automatically trigger workflows whenever code is pushed or updated, facilitating faster and more reliable model delivery.

## CircleCI Pipeline
The CircleCI pipeline configuration **(.circleci/config.yml)** includes steps for:
 -Installing dependencies
 -Running unit and integration tests
 -Training the ML model
 -Packaging and deployment

## How to Run Locally
1) Clone the repository:
2) Install dependencies:
3) Run the training script:
   -python application.py
