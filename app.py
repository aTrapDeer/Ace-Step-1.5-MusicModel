import os

# On Hugging Face Spaces Zero, `spaces` must be imported before CUDA-related modules.
if os.getenv("SPACE_ID"):
    try:
        import spaces  # noqa: F401
    except Exception:
        pass

from lora_ui import build_ui

app = build_ui()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
