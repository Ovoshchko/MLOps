# Inference

FastAPI-сервис для онлайн-предсказаний. Принимает «сырые» строки от источника, валидирует/проецирует их по манифесту датасета, считает фичи (включая текстовые эмбеддинги) и идёт в Triton за инференсом ONNX-модели.

```
client ──HTTP POST /predict──▶ FastAPI ──gRPC──▶ Triton (ONNX)
                                  │
                                  ├── manifest.json (из MLflow)
                                  └── SentenceTransformer (in-process)
```

## 1. Подъём инфраструктуры

Всё нужное описано в корневом `docker-compose.yml`. Поднимаются три сервиса, нужные для inference:


| Сервис   | Порт хоста                                   | Зачем нужен                                                                                               |
| -------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `minio`  | `9000` (API), `9001` (UI: admin/password123) | S3-совместимое хранилище под артефакты MLflow и parquet-датасеты.                                         |
| `mlflow` | `5001`                                       | Tracking server и артефакт-стор (хранит `manifest.json` обучающего датасета и зарегистрированные модели). |
| `triton` | `8001` (gRPC), `8002` (metrics)              | Inference-server, грузит модель из `inference/model_repository/`.                                         |


Из корня проекта:

```bash
docker compose up -d minio mlflow triton
```

Дождитесь, пока контейнеры станут `healthy`/`running`:

```bash
docker compose ps
docker compose logs -f triton   # должен закончиться "Started GRPCInferenceService at 0.0.0.0:8001"
```

> **Конфликт портов.** Triton по умолчанию публикует и `8000` (HTTP), и `8001` (gRPC). FastAPI ниже тоже слушает `8000`. Сервис ходит в Triton только по **gRPC** (`config.yml → triton.grpc_url: localhost:8001`), HTTP-порт Triton не используется. Если поднимаете локально и видите конфликт `8000` — оставьте только gRPC (можно убрать `8000:8000` из секции `triton` в `docker-compose.yml`) либо запускайте FastAPI на другом порту (см. ниже).

## 2. Подготовка модели и манифеста

В `inference/model_repository/regression/` лежит:

- `1/model.onnx` — экспорт обученного CatBoost (см. `experiments/train_experiment.py` → `log_catboost_model(format=["onnx"])`).
- `config.pbtxt` — Triton-конфиг (включён dynamic batching, `max_batch_size: 64`).

В `inference/src/config.yml`:

```yaml
mlflow_schema_path: mlflow-artifacts:/<exp>/<run_id>/artifacts/datasets/<dataset>/manifest.json
input_mapping: {}                 # source_field -> manifest_feature, если нужно
embedder:
  model_name: intfloat/multilingual-e5-small
  embedding_batch_size: 32
features:
  title_stats: false
triton:
  grpc_url: localhost:8001
  model_name: regression
```

Замените `mlflow_schema_path` на URI манифеста того датасета, на котором обучалась модель (из MLflow run, артефакт `datasets/<dataset>/manifest.json`).

## 3. Запуск FastAPI

Зависимости (помимо корневого `pyproject.toml`): `fastapi`, `uvicorn`, `tritonclient[grpc]`, `pydantic`, `sentence-transformers`. Установите в виртуальное окружение проекта.

Перед запуском нужны переменные для доступа к MLflow/MinIO (читаются `configure_mlflow()` и `mlflow.artifacts.download_artifacts`):

```bash
# bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=password123
```

```powershell
# PowerShell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "admin"
$env:AWS_SECRET_ACCESS_KEY = "password123"
```

Запуск (из корня проекта):

```bash
fastapi dev inference/src/app.py
```

Сервис стартует на `http://127.0.0.1:8000`. Документация — `http://127.0.0.1:8000/docs` (Swagger покажет конкретные поля `InputRow`, сгенерированные из манифеста).

Если порт `8000` занят Triton-HTTP:

```bash
fastapi dev inference/src/app.py --port 8010
```

## 4. Пример батч-запроса

Сервис принимает массив сырых строк и сам отбрасывает лишние поля по манифесту. NaN из pandas нужно превратить в `None`, иначе `requests` упадёт на сериализации (стандартный JSON не допускает `NaN`).

```python
import numpy as np
import pandas as pd
import requests

df = pd.read_parquet("load_date=2025-11-25.parquet").replace({np.nan: None})

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"rows": [df.iloc[i].to_dict() for i in range(64)]},
)
```

Что произойдёт под капотом:

1. **FastAPI/Pydantic** валидирует каждую строку против `InputRow` (типы из `manifest['schema']`). Лишние поля Twitter/Reddit (`author_premium`, `media_embed`, ...) молча отбрасываются благодаря `extra="ignore"`. Неверные типы дают `422` с указанием поля.
2. `**_build_dataframe`** заменяет `None` на дефолт по типу (`""`, `0.0`, `False`, `[]`) и собирает `pd.DataFrame` строго в порядке `manifest['features']`.
3. `**transform_features**` (общий с `data_pipeline`, см. `common/column_transforms.py`) применяет column-transforms из манифеста: `embed`, `log1p`, `coerce_bool`, `json_to_string`.
4. `**apply_feature_engineering**` добавляет engineered-фичи (`title_stats` и т.п.) согласно `config.features`.
5. **Triton** получает `[N, 398] float32` тензор `features` и возвращает `[N, 1]` `predictions`. Размер батча ограничен `max_batch_size: 64` в `config.pbtxt`.

Ответ:

```json
{ "predictions": [12.3, 4.7, 88.0, ...] }
```

## 5. Траблшутинг

- `Out of range float values are not JSON compliant: nan` — в payload остался NaN. Используйте `df.replace({np.nan: None})` перед `to_dict()`.
- `422 Unprocessable Entity` с указанием поля и причины — клиент прислал значение неверного типа. Поправьте источник или добавьте маппинг в `input_mapping` в `config.yml`.
- `model expects 2 dimensions (shape [-1, N]) but the model configuration specifies ...` в логах Triton — несоответствие числа фич между ONNX-графом и `config.pbtxt`. Сейчас `config.pbtxt` пользуется auto-complete (без явных `input/output`); если включаете явные `dims`, второе число должно совпадать с `N` ONNX-графа.
- `failed to connect to localhost:8001` — Triton не поднят или ещё стартует (`docker compose logs triton`).
- Сервис при старте уходит в `mlflow.artifacts.download_artifacts(...)` — если этот вызов висит, проверьте, что `MLFLOW_TRACKING_URI` указывает на `http://localhost:5001`, а MinIO доступен по `MLFLOW_S3_ENDPOINT_URL`.

