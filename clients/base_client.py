"""
Clase base abstracta: define la interfaz común que todo cliente debe cumplir.

Una clase abstracta (ABC) no se puede instanciar directamente. Sirve como
"contrato": cualquier clase que herede de BaseLLMClient está obligada a
implementar los métodos marcados con @abstractmethod. Así OpenAIClient y
AnthropicClient se usan exactamente de la misma forma.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):

    def __init__(self, api_key: str, config: ModelConfig):
        self.api_key = api_key
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Pide la respuesta completa al modelo y la devuelve de una sola vez."""

    @abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Devuelve la respuesta de a pedacitos (generador asíncrono con yield)."""

    @abstractmethod
    async def close(self) -> None:
        """Cierra la conexión con la API."""
