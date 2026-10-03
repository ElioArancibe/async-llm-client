"""Modelos Pydantic para mensajes, configuración, respuestas y errores."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de entrada del modelo, validados antes de llamar a la API."""

    model_config = ConfigDict(frozen=True)

    # None = usar el modelo por defecto del proveedor.
    model: str | None = None
    # None = no enviar temperatura y dejar el valor por defecto del modelo.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, gt=0)
    timeout: float = Field(default=60.0, gt=0)
    # Reintentos automáticos del SDK ante 429, 5xx y errores de red.
    max_retries: int = Field(default=2, ge=0, le=10)


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ErrorType(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    SERVER = "server"
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    """Error controlado que reemplaza a la excepción del SDK."""

    provider: Provider
    type: ErrorType
    message: str
    status_code: int | None = None
    retryable: bool = False

    def __str__(self) -> str:
        code = f" {self.status_code}" if self.status_code else ""
        return f"[{self.provider} {self.type}{code}] {self.message}"


class ModelResponse(BaseModel):
    """Resultado de una llamada no streaming: contenido o error, nunca una excepción."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
