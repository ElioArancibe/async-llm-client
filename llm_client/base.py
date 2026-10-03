"""Interfaz común para todos los proveedores."""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Provider


class MissingAPIKeyError(ValueError):
    """Falta la clave de API: es un error de configuración, se detecta al crear el cliente."""

    def __init__(self, env_var: str) -> None:
        super().__init__(f"Falta la variable de entorno {env_var}. Copiá .env.example a .env y completala.")
        self.env_var = env_var


class BaseLLMClient(ABC):
    """Clase base abstracta.

    Las subclases implementan `_generate`, `_stream` y `_to_error`. Los métodos
    públicos `generate` y `stream` capturan las excepciones del SDK y las
    convierten en un `LLMError`, así un error de API nunca rompe el loop principal.
    """

    provider: ClassVar[Provider]
    default_model: ClassVar[str]
    api_key_env: ClassVar[str]
    # Excepciones del SDK que se convierten en LLMError.
    sdk_errors: ClassVar[tuple[type[Exception], ...]]

    def __init__(self, config: ModelConfig | None = None, api_key: str | None = None) -> None:
        self.config = config or ModelConfig()
        self.model = self.config.model or self.default_model
        self.api_key = api_key or os.getenv(self.api_key_env)
        if not self.api_key:
            raise MissingAPIKeyError(self.api_key_env)

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Devuelve la respuesta completa del modelo (o un error controlado)."""
        try:
            return await self._generate(messages)
        except self.sdk_errors as exc:
            return ModelResponse(provider=self.provider, model=self.model, error=self._to_error(exc))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str | LLMError]:
        """Genera fragmentos de texto a medida que llegan.

        Si ocurre un error, el último elemento es un `LLMError` en lugar de una excepción.
        """
        try:
            async for chunk in self._stream(messages):
                yield chunk
        except self.sdk_errors as exc:
            yield self._to_error(exc)

    @abstractmethod
    async def _generate(self, messages: Sequence[ChatMessage]) -> ModelResponse: ...

    @abstractmethod
    def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...

    @abstractmethod
    def _to_error(self, exc: Exception) -> LLMError: ...

    @abstractmethod
    async def close(self) -> None: ...
