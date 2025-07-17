# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any


class Tracking(object):
    supported_backend = ["wandb", "mlflow", "console"]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: Union[str, List[str]] = "console",
        config=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn(
                    "`tracking` logger is deprecated. use `wandb` instead.",
                    DeprecationWarning,
                )
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._descriptions_logged = (
            set()
        )  # Track which descriptions we've already logged

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb
            import os

            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import mlflow

            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            # ? local makes sense since we use both tat and wandb
            # print('MATAN, DEBUG: I am surprised that `"console" in default_backend`')
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

    def log(self, data, step, backend=None):
        """
        Enhanced log method that supports metric descriptions.

        If data contains a '_descriptions' key, those descriptions will be logged
        as metadata to help document what each metric means.
        """
        # Extract descriptions if present
        descriptions = data.pop("_descriptions", {}) if isinstance(data, dict) else {}

        # Log descriptions to wandb summary (one-time setup for new metrics)
        if descriptions and "wandb" in self.logger:
            new_descriptions = {
                k: v
                for k, v in descriptions.items()
                if k not in self._descriptions_logged
            }
            if new_descriptions:
                # Store descriptions in wandb run summary for documentation
                wandb_instance = self.logger["wandb"]
                for metric_name, description in new_descriptions.items():
                    summary_key = f"_metric_descriptions/{metric_name}"
                    wandb_instance.summary[summary_key] = description
                    self._descriptions_logged.add(metric_name)

                # Also log as a table for better visibility
                if hasattr(wandb_instance, "Table"):
                    desc_table_data = [[k, v] for k, v in new_descriptions.items()]
                    desc_table = wandb_instance.Table(
                        data=desc_table_data, columns=["Metric", "Description"]
                    )
                    wandb_instance.log(
                        {f"_metric_descriptions_table_step_{step}": desc_table},
                        step=step,
                    )

        # Log the actual metrics
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if default_backend == "console" and descriptions:
                    # For console, print descriptions alongside metrics (only for new ones)
                    new_descriptions = {
                        k: v
                        for k, v in descriptions.items()
                        if k not in self._descriptions_logged
                    }
                    if new_descriptions:
                        print(f"\n--- Metric Descriptions (Step {step}) ---")
                        for metric_name, description in new_descriptions.items():
                            if metric_name in data:
                                print(f"{metric_name}: {description}")
                        print("--- End Descriptions ---\n")

                logger_instance.log(data=data, step=step)


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(
        _transform_params_to_json_serializable(params, convert_list_to_dict=True),
        sep="/",
    )


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(
        _transform_params_to_json_serializable,
        convert_list_to_dict=convert_list_to_dict,
    )

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {
                f"{i}": _transform(v) for i, v in enumerate(x)
            }
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans
