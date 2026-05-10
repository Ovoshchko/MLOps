from abc import ABC, abstractmethod
from typing import Any, Union

import mlflow
from catboost import CatBoostRegressor


class ONNXConverter(ABC):
    @abstractmethod
    def convert(self, model: Any, input_example: Any) -> Any:
        pass


class CatBoostONNXConverter(ONNXConverter):
    def __init__(self, model: Union[str, Any]):
        self.model = None

        if isinstance(model, str):
            self.model = self._load_model(model)
        else:
            self.model = model

    def _load_model(self, model_path: str) -> Any:
        if model_path.startswith("runs:/"):
            return self._load_model_from_mlflow(model_path)
        else:
            return CatBoostRegressor.load_model(model_path)

    def _load_model_from_mlflow(self, model_path: str) -> Any:
        run_id = model_path.split("/")[-2]
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        return model.get_raw_model()
    
    def convert(self, save_path: str) -> None:
        self.model.save_model(
            save_path,
            format="onnx",
            export_parameters={
                'onnx_domain': 'ai.catboost',
                'onnx_model_version': 1,
                'onnx_doc_string': 'Model for Regression',
                'onnx_graph_name': 'CatBoostModel_for_Regression'
            }
        )
