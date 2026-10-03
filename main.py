"""Script de validación: pregunta "¿Qué es la entropía?" en modo normal y en streaming.

Uso:
    python main.py                      # usa LLM_PROVIDER del .env
    python main.py --provider anthropic
"""

import argparse
import asyncio

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, LLMError, Provider, Role

QUESTION = "¿Qué es la entropía? Respondé en un párrafo corto."


async def run(provider: str | None) -> None:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="Sos un profesor de física claro y conciso."),
        ChatMessage(role=Role.USER, content=QUESTION),
    ]

    async with AsyncLLMManager.from_env(provider) as llm:
        print(f"Proveedor: {llm.provider} | Modelo: {llm.model}\n")

        print("=== Modo normal ===")
        response = await llm.generate(messages)
        if response.ok:
            print(response.content)
            if response.usage:
                print(
                    f"\n(tokens: {response.usage.input_tokens} entrada / "
                    f"{response.usage.output_tokens} salida, fin: {response.finish_reason})"
                )
        else:
            print(f"Error controlado: {response.error}")

        print("\n=== Modo streaming ===")
        async for chunk in llm.stream(messages):
            if isinstance(chunk, LLMError):
                print(f"\nError controlado: {chunk}")
                break
            print(chunk, end="", flush=True)
        print()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Prueba del cliente LLM asíncrono.")
    parser.add_argument(
        "--provider",
        choices=[p.value for p in Provider],
        help="Sobrescribe LLM_PROVIDER del .env",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args.provider))
    except ValueError as exc:
        # Errores de configuración: MissingAPIKeyError o un LLM_PROVIDER inválido.
        raise SystemExit(f"Error de configuración: {exc}") from None


if __name__ == "__main__":
    main()
