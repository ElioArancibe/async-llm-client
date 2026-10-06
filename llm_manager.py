"""
AsyncLLMManager: carga OpenAI o Anthropic según el proveedor elegido.

Usa el patrón Factory: el resto del programa no crea los clientes
directamente, sino que le pide al manager "dame el cliente de este proveedor".
Si mañana se agrega otro proveedor, solo hay que tocar este archivo.
"""

from collections.abc import AsyncIterator

from clients.anthropic_client import AnthropicClient
from clients.base_client import BaseLLMClient
from clients.gemini_client import GeminiClient
from clients.openai_client import OpenAIClient
from config import load_provider_settings
from schemas import ChatMessage, ModelConfig, ModelResponse, ProviderName, ProviderSettings


class AsyncLLMManager:

    def __init__(
        self,
        provider: ProviderName,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ):
        # 1. Leemos la API key y el modelo del proveedor desde el .env
        settings = load_provider_settings(provider)

        # 2. Validamos los parámetros del modelo con Pydantic
        #    (si temperature está fuera de 0..2, acá se lanza un ValidationError)
        config = ModelConfig(
            model=settings.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.provider = settings.provider
        self.model = settings.model

        # 3. Creamos el cliente que corresponde (Factory)
        self.client = self._create_client(settings, config)

    def _create_client(self, settings: ProviderSettings, config: ModelConfig) -> BaseLLMClient:
        # get_secret_value() saca el texto real de la API key guardada como SecretStr
        api_key = settings.api_key.get_secret_value()

        if settings.provider == "openai":
            return OpenAIClient(api_key, config)

        if settings.provider == "anthropic":
            return AnthropicClient(api_key, config)

        if settings.provider == "gemini":
            return GeminiClient(api_key, config)

        raise ValueError(f"Proveedor no soportado: {settings.provider}")

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa (modo normal)."""
        return await self.client.generate(messages)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Respuesta de a pedacitos (modo streaming)."""
        async for chunk in self.client.stream(messages):
            yield chunk

    async def close(self) -> None:
        await self.client.close()
