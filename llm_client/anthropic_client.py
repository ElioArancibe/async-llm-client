from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .schemas import (
    ChatMessage,
    ErrorType,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    Role,
    TokenUsage,
)


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC
    default_model = "claude-opus-5-5"
    api_key_env = "ANTHROPIC_API_KEY"
    sdk_errors = (anthropic.AnthropicError,)

    def __init__(self, config: ModelConfig | None = None, api_key: str | None = None) -> None:
        super().__init__(config, api_key)
        self._client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def _params(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        # Anthropic recibe el prompt de sistema aparte, no dentro de `messages`.
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "messages": [m.model_dump(mode="json") for m in messages if m.role != Role.SYSTEM],
        }
        if system:
            params["system"] = system
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def _generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        response = await self._client.messages.create(**self._params(messages))
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            provider=self.provider,
            model=response.model,
            content=text,
            finish_reason=response.stop_reason,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    def _to_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        match exc:
            case anthropic.AuthenticationError() | anthropic.PermissionDeniedError():
                kind, retryable = ErrorType.AUTHENTICATION, False
            case anthropic.RateLimitError():
                kind, retryable = ErrorType.RATE_LIMIT, True
            case anthropic.APITimeoutError():  # subclase de APIConnectionError: va antes
                kind, retryable = ErrorType.TIMEOUT, True
            case anthropic.APIConnectionError():
                kind, retryable = ErrorType.CONNECTION, True
            case anthropic.NotFoundError():
                kind, retryable = ErrorType.NOT_FOUND, False
            case anthropic.BadRequestError() | anthropic.UnprocessableEntityError():
                kind, retryable = ErrorType.BAD_REQUEST, False
            case anthropic.InternalServerError():  # incluye 529 overloaded
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
