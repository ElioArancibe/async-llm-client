import os
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Self

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Provider


class AsyncLLMManager:
    """Punto de entrada único: elige el cliente según el proveedor configurado."""

    _clients: dict[Provider, type[BaseLLMClient]] = {
        Provider.OPENAI: OpenAIClient,
        Provider.ANTHROPIC: AnthropicClient,
    }

    def __init__(
        self,
        provider: Provider | str,
        config: ModelConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = Provider(provider)  # ValueError si el proveedor no existe
        self.client = self._clients[self.provider](config, api_key)

    @classmethod
    def from_env(cls, provider: Provider | str | None = None) -> Self:
        """Construye el manager a partir de las variables de entorno.

        LLM_PROVIDER elige el proveedor (si no se pasa `provider`) y
        OPENAI_MODEL / ANTHROPIC_MODEL sobrescriben el modelo por defecto.
        """
        provider = Provider(provider or os.getenv("LLM_PROVIDER", Provider.OPENAI))
        model = os.getenv(f"{provider.upper()}_MODEL") or None
        return cls(provider, ModelConfig(model=model))

    @property
    def model(self) -> str:
        return self.client.model

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str | LLMError]:
        return self.client.stream(messages)

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
