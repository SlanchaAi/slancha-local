from slancha_local.backends.base import Backend, BackendCapability, BackendModel
from slancha_local.backends.ollama import OllamaBackend
from slancha_local.backends.registry import BackendRegistry

__all__ = ["Backend", "BackendCapability", "BackendModel", "BackendRegistry", "OllamaBackend"]
