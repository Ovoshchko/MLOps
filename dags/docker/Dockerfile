FROM apache/airflow:2.10.2

USER root
COPY dags/docker/requirements.txt /tmp/data_pipeline_requirements.txt
USER airflow

RUN pip install --no-cache-dir -r /tmp/data_pipeline_requirements.txt
