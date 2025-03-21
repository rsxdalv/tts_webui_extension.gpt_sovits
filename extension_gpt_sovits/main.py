import gradio as gr

def extension__tts_generation_webui():
    gpt_sovits_ui()
    return {
        "package_name": "extension_gpt_sovits",
        "name": "GPT-SoVITS",
        "version": "0.1.0",
        "requirements": "https://github.com/rsxdalv/GPT-SoVITS/releases/download/v2.6.3/gpt_sovits-2.6.3-py3-none-any.whl",
        "description": "GPT-SoVITS: A TTS solution powered by GPT and SoftVC VITS Singing Voice Conversion.",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "rsxdalv",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/X-T-E-R/GPT-SoVITS-Inference",
        "extension_website": "https://github.com/rsxdalv/extension_gpt_sovits",
        "extension_platform_version": "0.1.0",
    }

def gpt_sovits_ui():
    from extension_gpt_sovits.gradio_app import ui_core
    ui_core()

if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        extension__tts_generation_webui()
    demo.launch(
        server_port=7770,
    )