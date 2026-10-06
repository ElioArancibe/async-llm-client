# Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI**, **Anthropic** y **Gemini** en Python 3.12. Permite usar cualquiera de los tres proveedores con la misma interfaz, en modo normal o en streaming, con validación de datos mediante Pydantic y manejo de errores sin que el programa se rompa.

## Estructura del proyecto

```
async-llm-client/
├── main.py                  # Script de validación: elige proveedor y prueba normal + streaming
├── console.py               # Menú por consola para elegir el proveedor
├── llm_runner.py            # LLMRunner: ejecuta un prompt con un proveedor y muestra el resultado
├── llm_manager.py           # AsyncLLMManager: crea el cliente correcto (patrón Factory)
├── config.py                # Lee las API keys y los modelos desde las variables de entorno
├── schemas.py               # Esquemas Pydantic: ChatMessage, ModelConfig, ProviderSettings, ModelResponse
├── clients/
│   ├── base_client.py       # BaseLLMClient: clase base abstracta (interfaz común)
│   ├── openai_client.py     # OpenAIClient: usa AsyncOpenAI
│   ├── anthropic_client.py  # AnthropicClient: usa AsyncAnthropic
│   └── gemini_client.py     # GeminiClient: usa google-genai (client.aio)
├── requirements.txt
└── .env.example
```

### Cómo se conectan las piezas

```
main.py
  └─ console.choose_provider()          → el usuario elige "openai", "anthropic" o "gemini"
  └─ LLMRunner(provider, prompt)
       └─ AsyncLLMManager(provider)
            ├─ config.load_provider_settings()            → API key y modelo desde el .env
            ├─ ModelConfig(...)                            → validación de parámetros con Pydantic
            └─ OpenAIClient, AnthropicClient o GeminiClient → los tres heredan de BaseLLMClient
```

## Instalación

Requiere **Python 3.12**.

```bash
# 1. Crear el entorno virtual
py -3.12 -m venv .venv            # Windows
python3.12 -m venv .venv          # Linux / macOS

# 2. Activarlo
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

# 3. Instalar las dependencias (openai, anthropic, google-genai, pydantic, python-dotenv)
pip install -r requirements.txt
```

## Variables de entorno

Copiá `.env.example` como `.env` y completá tus claves:

```bash
copy .env.example .env            # Windows
cp .env.example .env              # Linux / macOS
```

| Variable            | Obligatoria                         | Descripción                   |
| ------------------- | ----------------------------------- | ----------------------------- |
| `OPENAI_API_KEY`    | Si elegís OpenAI                    | Clave de API de OpenAI        |
| `ANTHROPIC_API_KEY` | Si elegís Anthropic                 | Clave de API de Anthropic     |
| `GEMINI_API_KEY`    | Si elegís Gemini                    | Clave de API de Gemini        |
| `OPENAI_MODEL`      | No (por defecto `gpt-5-mini`)       | Modelo de OpenAI a usar       |
| `ANTHROPIC_MODEL`   | No (por defecto `claude-opus-5-5`)  | Modelo de Anthropic a usar    |
| `GEMINI_MODEL`      | No (por defecto `gemini-3.8-flash`) | Modelo de Gemini a usar       |

El proveedor **no** se configura en el `.env`: se elige por consola al ejecutar el programa.

El archivo `.env` está en `.gitignore` y nunca se sube al repositorio.

## Ejecutar el script de prueba

```bash
python main.py
```

El programa pregunta qué proveedor usar:

```
¿Qué proveedor querés usar?
  1) OpenAI
  2) Anthropic
  3) Gemini
Elegí una opción (1, 2 o 3): 1
```

Después hace la pregunta *"¿Qué es la entropía?"* dos veces:

1. **Modo normal:** espera la respuesta completa y la muestra de una vez, junto con los tokens usados.
2. **Modo streaming:** muestra el texto a medida que llega de la API.

## Cómo se cumple cada requisito

| Requisito | Dónde |
| --- | --- |
| Interfaz común para OpenAI, Anthropic y Gemini | `BaseLLMClient` en `clients/base_client.py`. Los tres clientes heredan de ella y tienen los mismos métodos: `generate()`, `stream()` y `close()`. |
| `AsyncLLMManager` que carga el proveedor según la configuración | `llm_manager.py` (patrón Factory). |
| Llamadas no bloqueantes | Se usan `AsyncOpenAI`, `AsyncAnthropic` y `client.aio` de `google-genai` con `await`, y `asyncio.run()` como punto de entrada. |
| Streaming con `yield` | El método `stream()` de cada cliente recorre el stream del SDK con `async for` y hace `yield` de cada fragmento. |
| Validación con Pydantic | `schemas.py`: `ModelConfig` valida `temperature` (de 0 a 2), `max_tokens` (> 0), `timeout_seconds` y `max_retries`. `ChatMessage` valida el rol y que el contenido no esté vacío. |
| Secretos protegidos | La API key se guarda como `SecretStr`, así que no aparece si se imprime la configuración. |
| Errores de red y rate limit controlados | Cada cliente captura las excepciones del SDK y las traduce a un mensaje claro. Ver la sección siguiente. |

### Manejo de errores

- **Modo normal:** `generate()` nunca lanza una excepción por un error de la API. Devuelve un `ModelResponse` con `success=False` y el motivo en `error`.
- **Modo streaming:** `stream()` termina con un último fragmento `[ERROR] ...`.
- **Errores que se capturan:**
  - API key inválida
  - Límite de tasa o de cuota (429)
  - Error de red
  - Timeout: `asyncio.timeout()` corta la espera si el modelo tarda demasiado
  - Cualquier otro error de la API
- **Reintentos:** ante errores temporales (429, errores 5xx, fallas de red) los SDKs de OpenAI y Anthropic reintentan automáticamente con espera exponencial; la cantidad se configura con `max_retries` en `ModelConfig`. El SDK de Gemini (`google-genai`) no expone esta opción, así que `GeminiClient` no reintenta solo: si falla, el error se reporta como siempre.
- **Errores de configuración:** si falta la API key, el programa lo avisa con un mensaje claro y termina sin traceback.

### Nota sobre la temperatura

`temperature` se valida entre 0 y 2, pero solo se envía a la API si se define. Algunos modelos recientes, como la familia GPT-5 de OpenAI y los Claude más nuevos, no aceptan una temperatura personalizada. Si se envía igual, la API responde con un error, que el programa muestra como mensaje controlado.
