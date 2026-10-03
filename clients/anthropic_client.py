"""Cliente asíncrono para Anthropic (Claude)."""

import asyncio
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from clients.base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):

    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(api_key, config)

        # Usamos AsyncAnthropic (la versión ASÍNCRONA del SDK) para no bloquear el event loop.
        self.client = AsyncAnthropic(
            api_key=api_key,
            # Reintentos automáticos ante 429 (rate limit), errores 5xx y fallas de red
            max_retries=config.max_retries,
            # Tiempo máximo de espera de cada request (también aplica al streaming)
            timeout=config.timeout_seconds,
        )

    def _build_request(self, messages: list[ChatMessage]) -> dict:
        """Arma los parámetros que espera la API de Anthropic."""

        # Diferencia con OpenAI: Anthropic no acepta mensajes con role="system"
        # dentro de la lista. El mensaje de sistema va en un parámetro aparte.
        system_text = ""
        conversation = []
        for message in messages:
            if message.role == "system":
                system_text = message.content
            else:
                conversation.append(message.model_dump())

        request = {
            "model": self.config.model,
            "messages": conversation,
            "max_tokens": self.config.max_tokens,  # en Anthropic es obligatorio
        }

        if system_text:
            request["system"] = system_text

        # La temperatura solo se envía si se definió.
        # (Los modelos Claude más nuevos no aceptan valores personalizados.)
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature

        return request

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        request = self._build_request(messages)

        try:
            # asyncio.timeout corta la espera si el modelo tarda demasiado
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await self.client.messages.create(**request)

        except (anthropic.AnthropicError, TimeoutError) as error:
            # En lugar de dejar que la excepción rompa el programa,
            # devolvemos una respuesta con success=False y el motivo del error.
            return ModelResponse(
                provider="anthropic",
                model=self.config.model,
                success=False,
                error=describe_anthropic_error(error),
            )

        # En Anthropic, response.content es una LISTA de bloques.
        # Puede haber bloques de distinto tipo (por ejemplo "thinking"),
        # así que juntamos solo los de tipo "text".
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return ModelResponse(
            provider="anthropic",
            model=response.model,
            content=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        request = self._build_request(messages)

        try:
            # messages.stream abre una conexión que va enviando el texto de a partes
            async with self.client.messages.stream(**request) as stream:
                # text_stream entrega solo los pedacitos de texto, a medida que llegan
                async for text in stream.text_stream:
                    yield text

        except anthropic.AnthropicError as error:
            # Si falla en medio del streaming, avisamos con un último fragmento de error
            yield f"\n[ERROR] {describe_anthropic_error(error)}"

    async def close(self) -> None:
        await self.client.close()


def describe_anthropic_error(error: Exception) -> str:
    """Traduce una excepción de Anthropic a un mensaje claro en español."""

    if isinstance(error, TimeoutError):
        return "El modelo tardó demasiado en responder (timeout)."

    if isinstance(error, anthropic.AuthenticationError):
        return "API key inválida. Revisá ANTHROPIC_API_KEY en el archivo .env."

    if isinstance(error, anthropic.RateLimitError):
        return "Límite de tasa o de cuota excedido (error 429). Probá de nuevo más tarde."

    # APIConnectionError cubre errores de red (incluye el timeout del SDK)
    if isinstance(error, anthropic.APIConnectionError):
        return "Error de red: no se pudo conectar con Anthropic."

    # Cualquier otra respuesta de error de la API (400, 404, 500, 529, etc.)
    if isinstance(error, anthropic.APIStatusError):
        return f"Error de la API de Anthropic ({error.status_code}): {error.message}"

    return f"Error inesperado de Anthropic: {error}"
