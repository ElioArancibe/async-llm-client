"""
Script de validación.

1. Carga las variables del archivo .env
2. Pide elegir el proveedor por consola (OpenAI o Anthropic)
3. Pregunta "¿Qué es la entropía?" en modo normal y en modo streaming

Uso:
    python main.py
"""

import asyncio

from dotenv import load_dotenv

from console import choose_provider
from llm_runner import LLMRunner

PROMPT = "¿Qué es la entropía? Respondé en un párrafo corto."


async def main() -> None:
    # Carga las variables del archivo .env (API keys) en el entorno
    load_dotenv()

    provider = choose_provider()

    try:
        runner = LLMRunner(provider=provider, prompt=PROMPT)
    except ValueError as error:
        # Errores de configuración: falta la API key, parámetros inválidos, etc.
        print(f"\nError de configuración: {error}")
        return

    print(f"\nProveedor: {runner.provider} | Modelo: {runner.manager.model}\n")

    try:
        await runner.run_normal()
        await runner.run_streaming()
    finally:
        # Cerramos la conexión aunque haya ocurrido un error
        await runner.close()


if __name__ == "__main__":
    # asyncio.run() es el punto de entrada: crea el event loop y ejecuta main()
    asyncio.run(main())
