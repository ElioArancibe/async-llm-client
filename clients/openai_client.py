"""Cliente asíncrono para OpenAI."""

import asyncio
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from clients.base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):

    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(api_key, config)

        # Usamos AsyncOpenAI (la versión ASÍNCRONA del SDK).
        # Si usáramos OpenAI (la síncrona) dentro de una función async,
        # bloquearíamos el event loop mientras el modelo "piensa".
        self.client = AsyncOpenAI(
            api_key=api_key,
            # Reintentos automáticos ante 429 (rate limit), errores 5xx y fallas de red
            max_retries=config.max_retries,
            # Tiempo máximo de espera de cada request (también aplica al streaming)
            timeout=config.timeout_seconds,
        )

    def _build_request(self, messages: list[ChatMessage]) -> dict:
        """Arma los parámetros que espera la API de OpenAI."""

        # Convertimos cada ChatMessage (objeto Pydantic) en un diccionario:
        # {"role": "user", "content": "..."}
        messages_as_dicts = []
        for message in messages:
            messages_as_dicts.append(message.model_dump())

        request = {
            "model": self.config.model,
            "messages": messages_as_dicts,
            # Los modelos actuales de OpenAI usan max_completion_tokens
            # en lugar del viejo max_tokens.
            "max_completion_tokens": self.config.max_tokens,
        }

        # La temperatura solo se envía si se definió.
        # (Algunos modelos, como la familia GPT-5, no aceptan valores personalizados.)
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature

        return request

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        request = self._build_request(messages)

        try:
            # asyncio.timeout corta la espera si el modelo tarda demasiado
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await self.client.chat.completions.create(**request)

        except (openai.OpenAIError, TimeoutError) as error:
            # En lugar de dejar que la excepción rompa el programa,
            # devolvemos una respuesta con success=False y el motivo del error.
            return ModelResponse(
                provider="openai",
                model=self.config.model,
                success=False,
                error=describe_openai_error(error),
            )

        # En OpenAI el texto está en choices[0].message.content
        text = response.choices[0].message.content or ""

        input_tokens = None
        output_tokens = None
        if response.usage is not None:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        return ModelResponse(
            provider="openai",
            model=response.model,
            content=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        request = self._build_request(messages)

        try:
            # Con stream=True la API no devuelve todo junto, sino un "stream" de chunks
            stream = await self.client.chat.completions.create(**request, stream=True)

            # async for recorre los chunks a medida que llegan por la red
            async for chunk in stream:
                # Cada chunk trae un pedacito de texto en choices[0].delta.content
                # (algunos chunks vienen vacíos, por eso el if)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except openai.OpenAIError as error:
            # Si falla en medio del streaming, avisamos con un último fragmento de error
            yield f"\n[ERROR] {describe_openai_error(error)}"

    async def close(self) -> None:
        await self.client.close()


def describe_openai_error(error: Exception) -> str:
    """Traduce una excepción de OpenAI a un mensaje claro en español."""

    if isinstance(error, TimeoutError):
        return "El modelo tardó demasiado en responder (timeout)."

    if isinstance(error, openai.AuthenticationError):
        return "API key inválida. Revisá OPENAI_API_KEY en el archivo .env."

    if isinstance(error, openai.RateLimitError):
        # OpenAI usa el mismo error 429 para dos casos distintos:
        # la cuenta se quedó sin crédito, o se hicieron demasiadas consultas seguidas.
        if error.type == "insufficient_quota":
            return "La cuenta de OpenAI no tiene crédito disponible (error 429). Cargá saldo en platform.openai.com."
        return "Límite de tasa excedido (error 429): demasiadas consultas seguidas. Probá de nuevo más tarde."

    # APIConnectionError cubre errores de red (incluye el timeout del SDK)
    if isinstance(error, openai.APIConnectionError):
        return "Error de red: no se pudo conectar con OpenAI."

    # Cualquier otra respuesta de error de la API (400, 404, 500, etc.)
    if isinstance(error, openai.APIStatusError):
        return f"Error de la API de OpenAI ({error.status_code}): {error.message}"

    return f"Error inesperado de OpenAI: {error}"
