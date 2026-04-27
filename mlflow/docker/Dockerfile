FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow==3.10.1 boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "s3://mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

