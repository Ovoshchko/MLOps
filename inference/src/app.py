from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict

from inference.src.kafka_consumer import KafkaInferenceConsumer
from inference.src.predictor import PredictionService


service = PredictionService.from_config_path(Path(__file__).parent / "config.yml")
InputRow = service.input_row_model


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rows: list[InputRow]


class PredictResponse(BaseModel):
    predictions: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.start()
    app.state.predictor = service
    app.state.kafka_consumer = None
    app.state.kafka_thread = None

    kafka_config = dict(service.config.get("kafka") or {})
    if bool(kafka_config.get("run_with_api", True)):
        kafka_consumer = KafkaInferenceConsumer(service)
        kafka_thread = Thread(target=kafka_consumer.run, daemon=True)
        kafka_thread.start()
        app.state.kafka_consumer = kafka_consumer
        app.state.kafka_thread = kafka_thread

    yield

    if app.state.kafka_consumer is not None:
        app.state.kafka_consumer.stop()
    if app.state.kafka_thread is not None:
        app.state.kafka_thread.join(timeout=10)


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    predictions = request.app.state.predictor.predict_rows(body.rows)
    return PredictResponse(predictions=predictions)
