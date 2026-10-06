"""
Esquemas de datos del proyecto, definidos con Pydantic.

Pydantic valida los datos en el momento en que se crea cada objeto:
si algo no cumple las reglas (por ejemplo, temperature=5), lanza un
ValidationError antes de llegar a llamar a la API.
"""

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

# Los únicos proveedores válidos del proyecto.
# Literal hace que Pydantic rechace cualquier otro texto.
ProviderName = Literal["openai", "anthropic", "gemini"]


class ChatMessage(BaseModel):
    """Un mensaje de la conversación: quién habla (role) y qué dice (content)."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)  # no se permiten mensajes vacíos


class ModelConfig(BaseModel):
    """Parámetros de entrada del modelo."""

    # Nombre del modelo, por ejemplo "gpt-5-mini" o "claude-opus-5-5".
    model: str

    # Temperatura entre 0 y 2. Es opcional: si queda en None no se envía
    # y el modelo usa su valor por defecto.
    temperature: float | None = Field(default=None, ge=0, le=2)

    # Cantidad máxima de tokens que puede generar el modelo (mayor a 0).
    max_tokens: int = Field(default=4096, gt=0)

    # Segundos máximos de espera por una respuesta.
    timeout_seconds: float = Field(default=60, gt=0)

    # Cuántas veces reintenta el SDK ante errores temporales (429, 5xx, red).
    max_retries: int = Field(default=2, ge=0, le=5)


class ProviderSettings(BaseModel):
    """Datos para conectarse a un proveedor, leídos del archivo .env."""

    provider: ProviderName
    # SecretStr oculta el valor al imprimirlo: print(settings) muestra
    # api_key=SecretStr('**********') en lugar de la clave real.
    api_key: SecretStr
    model: str


class ModelResponse(BaseModel):
    """Respuesta unificada: tiene la misma forma sin importar el proveedor."""

    provider: ProviderName
    model: str
    content: str = ""

    # Si algo falló, success es False y error explica qué pasó.
    success: bool = True
    error: str | None = None

    # Tokens consumidos (si la API los informa).
    input_tokens: int | None = None
    output_tokens: int | None = None
