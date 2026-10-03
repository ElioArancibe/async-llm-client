"""
Lectura de la configuración de cada proveedor desde las variables de entorno.

Las variables se cargan desde el archivo .env en main.py con load_dotenv().
"""

import os

from schemas import ProviderName, ProviderSettings

# Modelo que se usa si no se define OPENAI_MODEL / ANTHROPIC_MODEL en el .env
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-opus-5-5",
}

# Nombre de la variable de entorno que tiene la API key de cada proveedor
API_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Nombre de la variable de entorno (opcional) para cambiar el modelo
MODEL_VARIABLES = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}


def load_provider_settings(provider: ProviderName) -> ProviderSettings:
    """Devuelve la API key y el modelo del proveedor elegido."""

    api_key_variable = API_KEY_VARIABLES[provider]
    api_key = os.getenv(api_key_variable)

    if not api_key:
        raise ValueError(
            f"No se encontró la variable {api_key_variable}. "
            "Copiá .env.example como .env y completá tu API key."
        )

    model = os.getenv(MODEL_VARIABLES[provider])
    if not model:
        model = DEFAULT_MODELS[provider]

    # Pydantic valida los datos y guarda la API key como SecretStr
    return ProviderSettings(provider=provider, api_key=api_key, model=model)
