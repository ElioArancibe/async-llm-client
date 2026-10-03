"""
LLMRunner: ejecuta un prompt con el proveedor elegido y muestra el resultado.

Se crea con un proveedor y un prompt:
    runner = LLMRunner(provider="openai", prompt="¿Qué es la entropía?")
"""

from llm_manager import AsyncLLMManager
from schemas import ChatMessage, ProviderName


class LLMRunner:

    def __init__(self, provider: ProviderName, prompt: str):
        self.provider = provider

        # Convertimos el prompt (texto) en la lista de mensajes que esperan los clientes.
        # ChatMessage valida que el prompt no esté vacío.
        self.messages = [ChatMessage(role="user", content=prompt)]

        # El manager se encarga de crear el cliente correcto (OpenAI o Anthropic)
        self.manager = AsyncLLMManager(provider)

    async def run_normal(self) -> None:
        """Pide la respuesta completa y la muestra de una sola vez."""

        print("=== Modo normal ===")
        response = await self.manager.generate(self.messages)

        if response.success:
            print(response.content)
            print(f"\n(Tokens: {response.input_tokens} de entrada, {response.output_tokens} de salida)")
        else:
            # El error llega como dato, no como excepción: el programa sigue funcionando
            print(f"No se pudo obtener la respuesta: {response.error}")

        print()

    async def run_streaming(self) -> None:
        """Muestra la respuesta a medida que llega, pedacito por pedacito."""

        print("=== Modo streaming ===")

        async for chunk in self.manager.stream(self.messages):
            # end="" evita el salto de línea; flush=True muestra el texto en el momento
            print(chunk, end="", flush=True)

        print("\n")

    async def close(self) -> None:
        await self.manager.close()
