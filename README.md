# Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI** y **Anthropic** en Python 3.12: una única interfaz para generar respuestas completas o en streaming, con validación de entrada mediante Pydantic y manejo controlado de errores.

## Estructura

```
async-llm-client/
├── llm_client/
│   ├── schemas.py           # Modelos Pydantic: ChatMessage, ModelConfig, ModelResponse, LLMError
│   ├── base.py              # BaseLLMClient (clase base abstracta) y captura de errores
│   ├── openai_client.py     # OpenAIClient    (AsyncOpenAI)
│   ├── anthropic_client.py  # AnthropicClient (AsyncAnthropic)
│   └── manager.py           # AsyncLLMManager: elige el proveedor según la configuración
├── main.py                  # Script de validación (modo normal + streaming)
├── requirements.txt
└── .env.example
```

## Instalación

Requiere **Python 3.12**.

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Variables de entorno

Copiá `.env.example` como `.env` y completá los valores:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| Variable            | Obligatoria                        | Descripción                                          |
| ------------------- | ---------------------------------- | ---------------------------------------------------- |
| `LLM_PROVIDER`      | No (por defecto `openai`)          | Proveedor a usar: `openai` o `anthropic`             |
| `OPENAI_API_KEY`    | Si el proveedor es `openai`        | Clave de API de OpenAI                               |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic`     | Clave de API de Anthropic                            |
| `OPENAI_MODEL`      | No (por defecto `gpt-5-mini`)      | Modelo de OpenAI a usar                              |
| `ANTHROPIC_MODEL`   | No (por defecto `claude-opus-5-5`) | Modelo de Anthropic a usar                           |

El archivo `.env` está en `.gitignore`: **nunca se sube al repositorio**.

## Ejecutar el script de prueba

```bash
python main.py                        # usa LLM_PROVIDER del .env
python main.py --provider anthropic   # fuerza un proveedor
```

El script pregunta *"¿Qué es la entropía?"* dos veces: primero en modo normal, que imprime la respuesta completa y el uso de tokens, y después en modo streaming, que imprime el texto a medida que llega.

## Uso como librería

```python
import asyncio
from llm_client import AsyncLLMManager, ChatMessage, LLMError, ModelConfig, Role

async def demo():
    config = ModelConfig(max_tokens=2048)  # validado por Pydantic
    messages = [ChatMessage(role=Role.USER, content="Hola")]

    async with AsyncLLMManager("anthropic", config) as llm:
        # Modo normal
        response = await llm.generate(messages)
        print(response.content if response.ok else response.error)

        # Modo streaming
        async for chunk in llm.stream(messages):
            if isinstance(chunk, LLMError):
                print("Error:", chunk)
                break
            print(chunk, end="", flush=True)

asyncio.run(demo())
```

## Decisiones de diseño

- **Intercambiabilidad:** `OpenAIClient` y `AnthropicClient` heredan de `BaseLLMClient` y exponen los mismos métodos, `generate()` y `stream()`. `AsyncLLMManager` instancia uno u otro según `LLM_PROVIDER`.
- **Asincronía:** se usan únicamente los clientes asíncronos de los SDKs (`AsyncOpenAI`, `AsyncAnthropic`) con `await`, así que ninguna llamada bloquea el event loop.
- **Streaming:** `stream()` es un generador asíncrono que hace `yield` de cada fragmento de texto dentro de un `async for` sobre el stream del SDK.
- **Validación:** `ChatMessage` valida el rol y que el contenido no esté vacío. `ModelConfig` valida `temperature` (de 0 a 2), `max_tokens` (> 0), `timeout` y `max_retries`.
- **Manejo de errores:**
  - **Durante una llamada:** las excepciones del SDK (clave inválida, rate limit, errores de red o timeouts, errores del servidor) se capturan y se convierten en un `LLMError` con `type`, `status_code` y `retryable`. `generate()` lo devuelve dentro de `ModelResponse.error` y `stream()` lo emite como último elemento. Ninguno de los dos lanza una excepción.
  - **Reintentos:** los errores reintentables (429, 5xx, problemas de red) se reintentan automáticamente con backoff exponencial gracias a los SDKs. La cantidad se configura con `ModelConfig.max_retries`, por defecto 2.
  - **Al crear el cliente:** una clave faltante es un error de configuración y se detecta en ese momento con `MissingAPIKeyError`.
- **Temperatura:** solo se envía si se define en `ModelConfig`. Algunos modelos recientes, como la familia GPT-5 y Claude Opus 5.5, no aceptan valores personalizados. En ese caso la API responde con un error, que vuelve como `LLMError` de tipo `bad_request`.
