"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.model import CLSModel

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", "CLSModel", "check_sru_requirement"]
