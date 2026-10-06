"""Interacción con el usuario por consola."""


from schemas import ChatMessage, ProviderName


def choose_provider() -> ProviderName:
    """Muestra un menú y repite la pregunta hasta que se elija una opción válida."""

    print("¿Qué proveedor querés usar?")
    print("  1) OpenAI")
    print("  2) Anthropic")
    print("  3) Gemini")

    while True:
        option = input("Elegí una opción (1, 2 o 3): ").strip()

        if option == "1":
            return "openai"

        if option == "2":
            return "anthropic"

        if option == "3":
            return "gemini"

        print("Opción inválida. Escribí 1, 2 o 3.")

def get_console_prompt() -> str:
    while True:
        user_input = input("Escribí tu prompt: ").strip()

        if user_input:
            return user_input

        print("No puede estar vacío. Por favor, escribí algo.")
