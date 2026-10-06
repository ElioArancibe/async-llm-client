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

from console import choose_provider, get_console_prompt
from llm_runner import LLMRunner

async def main() -> None:
    # Carga las variables del archivo .env (API keys) en el entorno
    load_dotenv()

    # Elije el provedor por consola
    provider = choose_provider()
    # Pregunta el prompt por consola
    prompt = get_console_prompt()

    try:
        runner = LLMRunner(provider=provider, prompt=prompt)
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
