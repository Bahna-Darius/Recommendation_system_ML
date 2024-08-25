# Deploy AI model to Cloud

1. **Development**: The developer makes changes on their local machine.
2. **Upload Training Data**: All training data is uploaded to Google Cloud Storage (GCS).
3. **Push to GitHub**: The code changes are then pushed to GitHub.
4. **CI/CD Pipeline**: Once the changes are on GitHub, a CircleCI job is triggered, building the Docker image.
5. **Docker Image Storage**: At the end of the CircleCI job, the built Docker image is pushed to Google Cloud Container Registry.
6. **Model Training**: A job is then added to Google Cloud Vertex AI Training to train the model using the provided training data (retrieved from GCS).
7. **Training Completion**: Once the training job is complete, the model is stored in GCS.
8. **Model Registration**: With the model in GCS, it can then be imported into the Vertex AI Model Registry.
9. **Model Deployment**: Finally, the model can be deployed and exposed on an endpoint using Vertex AI Online Prediction.


