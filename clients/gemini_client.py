"""Cliente asíncrono para Google Gemini."""

import asyncio
from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from clients.base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class GeminiClient(BaseLLMClient):

    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(api_key, config)

        # Usamos client.aio (la versión ASÍNCRONA del SDK) para no bloquear el event loop.
        # attempts cuenta el primer intento, por eso se suma 1 a max_retries.
        # Así reintenta ante errores temporales (429, 503 por alta demanda, etc.)
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=config.max_retries + 1),
            ),
        )

    def _build_contents(self, messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        """Separa el mensaje de sistema del resto de la conversación.

        Igual que Anthropic, Gemini no acepta role="system" dentro de los
        mensajes: va aparte, en la configuración, como system_instruction.
        Además Gemini usa role="model" en lugar de "assistant".
        """

        system_text = ""
        contents = []
        for message in messages:
            if message.role == "system":
                system_text = message.content
            else:
                role = "model" if message.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": message.content}]})

        return system_text, contents

    def _build_config(self, system_text: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_text or None,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            # Desactivamos el "automatic function calling" (AFC): no usamos herramientas,
            # y si queda activo el SDK muestra un aviso en la consola.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        system_text, contents = self._build_contents(messages)
        generate_config = self._build_config(system_text)

        try:
            # asyncio.timeout corta la espera si el modelo tarda demasiado
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await self.client.aio.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=generate_config,
                )

        except (errors.APIError, httpx.HTTPError, TimeoutError) as error:
            # En lugar de dejar que la excepción rompa el programa,
            # devolvemos una respuesta con success=False y el motivo del error.
            return ModelResponse(
                provider="gemini",
                model=self.config.model,
                success=False,
                error=describe_gemini_error(error),
            )

        return ModelResponse(
            provider="gemini",
            model=response.model_version or self.config.model,
            content=response.text or "",
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        system_text, contents = self._build_contents(messages)
        generate_config = self._build_config(system_text)

        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=generate_config,
            )

            # async for recorre los chunks a medida que llegan por la red
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except (errors.APIError, httpx.HTTPError) as error:
            # Si falla en medio del streaming, avisamos con un último fragmento de error
            yield f"\n[ERROR] {describe_gemini_error(error)}"

    async def close(self) -> None:
        await self.client.aio.aclose()


def describe_gemini_error(error: Exception) -> str:
    """Traduce una excepción de Gemini a un mensaje claro en español."""

    if isinstance(error, TimeoutError):
        return "El modelo tardó demasiado en responder (timeout)."

    if isinstance(error, errors.APIError):
        message = error.message or str(error)

        # Gemini no siempre usa 401/403 para esto: una API key inválida
        # suele llegar como 400 INVALID_ARGUMENT con este mensaje puntual.
        if "api key not valid" in message.lower() or error.code in (401, 403):
            return "API key inválida. Revisá GEMINI_API_KEY en el archivo .env."

        if error.code == 429:
            return "Límite de tasa o de cuota excedido (error 429). Probá de nuevo más tarde."

        return f"Error de la API de Gemini ({error.code}): {message}"

    # A diferencia de OpenAI y Anthropic, el SDK de Gemini no envuelve los errores
    # de red: llegan como excepciones de httpx (la librería HTTP que usa por debajo).
    if isinstance(error, httpx.HTTPError):
        return "Error de red: no se pudo conectar con Gemini."

    return f"Error inesperado de Gemini: {error}"
