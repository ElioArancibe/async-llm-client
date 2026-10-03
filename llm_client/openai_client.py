from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .schemas import (
    ChatMessage,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    TokenUsage,
)


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI
    default_model = "gpt-5-mini"
    api_key_env = "OPENAI_API_KEY"
    sdk_errors = (openai.OpenAIError,)

    def __init__(self, config: ModelConfig | None = None, api_key: str | None = None) -> None:
        super().__init__(config, api_key)
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def _params(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(mode="json") for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def _generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        response = await self._client.chat.completions.create(**self._params(messages))
        choice = response.choices[0]
        usage = (
            TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            if response.usage
            else None
        )
        return ModelResponse(
            provider=self.provider,
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(**self._params(messages), stream=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _to_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        match exc:
            case openai.AuthenticationError() | openai.PermissionDeniedError():
                kind, retryable = ErrorType.AUTHENTICATION, False
            case openai.RateLimitError():
                kind, retryable = ErrorType.RATE_LIMIT, True
            case openai.APITimeoutError():  # subclase de APIConnectionError: va antes
                kind, retryable = ErrorType.TIMEOUT, True
            case openai.APIConnectionError():
                kind, retryable = ErrorType.CONNECTION, True
            case openai.NotFoundError():
                kind, retryable = ErrorType.NOT_FOUND, False
            case openai.BadRequestError() | openai.UnprocessableEntityError():
                kind, retryable = ErrorType.BAD_REQUEST, False
            case openai.InternalServerError():
                kind, retryable = ErrorType.SERVER, True
            case _:
                kind, retryable = ErrorType.UNKNOWN, False
        message = getattr(exc, "message", None) or str(exc)
        return LLMError(
            provider=self.provider,
            type=kind,
            message=message,
            status_code=status,
            retryable=retryable,
        )

    async def close(self) -> None:
        await self._client.close()
