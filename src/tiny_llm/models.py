from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from .qwen2_week3 import Qwen2ModelWeek3
from importlib import import_module
from .qwen3 import Qwen3Model


def _optional_qwen3_model(module_name: str, class_name: str):
    try:
        module = import_module(f"{__package__}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__}.{module_name}":
            return None
        raise
    return getattr(module, class_name)


Qwen3ModelWeek1 = _optional_qwen3_model("qwen3_week1", "Qwen3ModelWeek1")
Qwen3ModelWeek2 = _optional_qwen3_model("qwen3_week2", "Qwen3ModelWeek2") or Qwen3Model
Qwen3ModelWeek3 = _optional_qwen3_model("qwen3_week3", "Qwen3ModelWeek3")


def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen2-7b":
        return "Qwen/Qwen2-7B-Instruct-MLX"
    elif lower_shortcut_name == "qwen2-0.5b":
        return "Qwen/Qwen2-0.5B-Instruct-MLX"
    elif lower_shortcut_name == "qwen2-1.5b":
        return "Qwen/Qwen2-1.5B-Instruct-MLX"
    elif lower_shortcut_name == "qwen3-8b":
        return "mlx-community/Qwen3-8B-4bit"
    elif lower_shortcut_name == "qwen3-0.6b":
        return "mlx-community/Qwen3-0.6B-4bit"
    elif lower_shortcut_name == "qwen3-1.7b":
        return "mlx-community/Qwen3-1.7B-4bit"
    elif lower_shortcut_name == "qwen3-4b":
        return "mlx-community/Qwen3-4B-4bit"
    else:
        return shortcut_name


def dispatch_model(model_name: str, mlx_model, week: int, **kwargs):
    model_name = shortcut_name_to_full_name(model_name)
    is_qwen3 = model_name.startswith("Qwen/Qwen3") or model_name.startswith(
        "mlx-community/Qwen3"
    )
    if week == 1 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek1(mlx_model, **kwargs)
    elif week == 2 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek2(mlx_model, **kwargs)
    elif week == 3 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek3(mlx_model, **kwargs)
    elif week == 1 and is_qwen3 and Qwen3ModelWeek1 is not None:
        return Qwen3ModelWeek1(mlx_model, **kwargs)
    elif week == 2 and is_qwen3:
        return Qwen3ModelWeek2(mlx_model, **kwargs)
    elif week == 3 and is_qwen3 and Qwen3ModelWeek3 is not None:
        return Qwen3ModelWeek3(mlx_model, **kwargs)
    else:
        raise ValueError(f"{model_name} for week {week} not supported")
