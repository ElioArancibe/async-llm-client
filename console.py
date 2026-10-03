"""Interacción con el usuario por consola."""

from schemas import ProviderName


def choose_provider() -> ProviderName:
    """Muestra un menú y repite la pregunta hasta que se elija una opción válida."""

    print("¿Qué proveedor querés usar?")
    print("  1) OpenAI")
    print("  2) Anthropic")

    while True:
        option = input("Elegí una opción (1 o 2): ").strip()

        if option == "1":
            return "openai"

        if option == "2":
            return "anthropic"

        print("Opción inválida. Escribí 1 o 2.")
