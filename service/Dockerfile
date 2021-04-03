FROM python:3.7-slim
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./main.py .
COPY anomalyModel.joblib .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

