import gradio as gr
from gpt_sovits.Synthesizers.base import get_wave_header_chunk
from gpt_sovits.src.common_config_manager import app_config, __version__
from importlib import import_module
from functools import partial
import os
from huggingface_hub import HfApi, hf_hub_download
import nltk
import requests
import zipfile
import os
import urllib.parse

frontend_version = __version__

synthesizer_name = app_config.synthesizer

# Global variables to store components and state
all_gradio_components = {}
characters_and_emotions_dict = {}


def load_character_emotions(character_name, characters_and_emotions):
    emotion_options = characters_and_emotions.get(character_name, ["default"])
    return gr.Dropdown(emotion_options, value="default")


def get_audio(*data, streaming=False):
    # Map data to named parameters
    data = dict(zip([key for key in all_gradio_components.keys()], data))
    data["stream"] = streaming

    # Create TTS synthesizer instance
    # Dynamically import synthesizer module
    synthesizer_module = import_module(f"gpt_sovits.Synthesizers.{synthesizer_name}")
    TTS_Synthesizer = synthesizer_module.TTS_Synthesizer

    # Create synthesizer instance
    tts_synthesizer = TTS_Synthesizer(debug_mode=True)

    if data.get("text") in ["", None]:
        gr.Warning("Text cannot be empty")
        return None, None
    try:
        task = tts_synthesizer.params_parser(data)

        if not streaming:
            if synthesizer_name == "remote":
                save_path = tts_synthesizer.generate(task, return_type="filepath")
                yield save_path
            else:
                gen = tts_synthesizer.generate(task, return_type="numpy")
                yield next(gen)
        else:
            gen = tts_synthesizer.generate(task, return_type="numpy")
            sample_rate = 32000 if task.sample_rate in [None, 0] else task.sample_rate
            yield get_wave_header_chunk(sample_rate=sample_rate)
            for chunk in gen:
                yield chunk

    except Exception as e:
        gr.Warning(f"Error: {e}")


get_streaming_audio = partial(get_audio, streaming=True)


def get_characters_and_emotions():
    global characters_and_emotions_dict

    # Get the synthesizer
    synthesizer_module = import_module(f"gpt_sovits.Synthesizers.{synthesizer_name}")
    TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
    tts_synthesizer = TTS_Synthesizer(debug_mode=True)

    # Check if dictionary is empty to avoid duplicate fetches
    if characters_and_emotions_dict == {}:
        characters_and_emotions_dict = tts_synthesizer.get_characters()
        print(characters_and_emotions_dict)

    return characters_and_emotions_dict


def change_character_list(character="", emotion="default"):
    characters_and_emotions = {}

    try:
        characters_and_emotions = get_characters_and_emotions()
        character_names = [i for i in characters_and_emotions]
        if len(character_names) != 0:
            if character in character_names:
                character_name_value = character
            else:
                character_name_value = character_names[0]
        else:
            character_name_value = ""
        emotions = characters_and_emotions.get(character_name_value, ["default"])
        emotion_value = emotion

    except:
        character_names = []
        character_name_value = ""
        emotions = ["default"]
        emotion_value = "default"
        characters_and_emotions = {}

    return (
        gr.Dropdown(
            character_names, value=character_name_value, label="Select Character"
        ),
        gr.Dropdown(
            emotions, value=emotion_value, label="Emotion List", interactive=True
        ),
        characters_and_emotions,
    )


def cut_sentence_multilang(text, max_length=30):
    if max_length == -1:
        return text, ""
    # Initialize counter
    word_count = 0
    in_word = False

    for index, char in enumerate(text):
        if char.isspace():  # If current character is space
            in_word = False
        elif (
            char.isascii() and not in_word
        ):  # If ASCII character (English) and not in word
            word_count += 1  # New English word
            in_word = True
        elif not char.isascii():  # If non-English character
            word_count += 1  # Each non-English character counts as one word
        if word_count > max_length:
            return text[:index], text[index:]

    return text, ""


def initialize_synthesizer():
    # Dynamically import synthesizer module
    synthesizer_module = import_module(f"gpt_sovits.Synthesizers.{synthesizer_name}")
    TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
    TTS_Task = synthesizer_module.TTS_Task

    # Create synthesizer instance
    tts_synthesizer = TTS_Synthesizer(debug_mode=True)
    tts_task_example = TTS_Task()

    return tts_synthesizer, tts_task_example



def download_gpt_sovits_models():
    try:
        import os
        import nltk

        os.makedirs("data/models/gpt_sovits", exist_ok=True)
        os.makedirs("data/models/gpt_sovits/prompt_cache", exist_ok=True)

        yield gr.Button("Downloading...", interactive=False), "Initializing NLTK..."
        nltk.download("punkt")

        yield gr.Button("Downloading...", interactive=False), "Downloading NLTK resources..."
        nltk.download("averaged_perceptron_tagger_eng")

        repo_id = "lj1995/GPT-SoVITS"
        destination_dir = "data/models/gpt_sovits/"

        yield gr.Button("Downloading...", interactive=False), "Creating directories..."
        os.makedirs(destination_dir, exist_ok=True)

        yield gr.Button("Downloading...", interactive=False), "Fetching file list from Hugging Face..."
        api = HfApi()
        repo_files = api.list_repo_files(repo_id)

        total_files = len(repo_files)
        for idx, file_name in enumerate(repo_files, 1):
            yield gr.Button("Downloading...", interactive=False), f"Downloading file {idx}/{total_files}: {file_name}"
            hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=destination_dir,
            )

        yield gr.Button("Download GPT-SoVITS Models", interactive=True), "Download completed successfully!"
        return gr.Button("Download GPT-SoVITS Models", interactive=True), "Download completed successfully!"
    except Exception as e:
        error_message = f"Error during download: {str(e)}"
        yield gr.Button("Download GPT-SoVITS Models", interactive=True), error_message
        return gr.Button("Download GPT-SoVITS Models", interactive=True), error_message


def download_trained_models():
    try:
        import os
        os.makedirs("data/models/gpt_sovits/trained", exist_ok=True)

        hf_links = [""] * 10
        hf_links[0] = "https://huggingface.co/baicai1145/GPT-SoVITS-STAR/resolve/main/%E9%93%B6%E7%8B%BC.zip?download=true"
        hf_links[1] = "https://huggingface.co/baicai1145/GPT-SoVITS-STAR/resolve/main/%E9%9D%92%E9%9B%80.zip?download=true"

        def get_decoder(file_name: str):
            try:
                return file_name.encode("cp437").decode("gbk")
            except:
                return file_name

        for i, hf_link in enumerate(hf_links):
            if hf_link == "":
                continue

            character_name = urllib.parse.unquote(os.path.basename(hf_link).rsplit(".", 1)[0])
            yield gr.Button("Downloading...", interactive=False), f"Downloading {character_name}..."

            destination_dir = "data/models/gpt_sovits/"
            output_path = os.path.join(destination_dir, "trained", character_name)
            os.makedirs(output_path, exist_ok=True)
            zip_file_path = os.path.join(output_path, "file.zip")

            response = requests.get(hf_link)
            with open(zip_file_path, "wb") as file:
                file.write(response.content)

            yield gr.Button("Downloading...", interactive=False), f"Extracting {character_name}..."
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    encoded_file_name = get_decoder(file_info.filename)
                    new_path = os.path.join(output_path, encoded_file_name)

                    if encoded_file_name[-1] not in ["\\", "/"]:
                        with open(new_path, "wb") as file:
                            file.write(zip_ref.read(file_info.filename))
                    else:
                        os.makedirs(new_path, exist_ok=True)

            os.remove(zip_file_path)
            yield gr.Button("Downloading...", interactive=False), f"{character_name} downloaded and extracted successfully!"

        yield gr.Button("Download Trained SoVITS Models", interactive=True), "All trained models downloaded successfully!"
        return gr.Button("Download Trained SoVITS Models", interactive=True), "All trained models downloaded successfully!"
    except Exception as e:
        error_message = f"Error during download: {str(e)}"
        yield gr.Button("Download Trained SoVITS Models", interactive=True), error_message
        return gr.Button("Download Trained SoVITS Models", interactive=True), error_message


def download_tab():
    with gr.Tab(label="Download Models"):
        gr.Markdown("## Download and Setup Models")

        with gr.Column():
            gr.Markdown("### GPT-SoVITS Model Download")
            download_gptsovits_btn = gr.Button("Download GPT-SoVITS Models", variant="primary")
            download_status = gr.Textbox(label="Download Status", interactive=False)

            download_gptsovits_btn.click(
                fn=download_gpt_sovits_models,
                outputs=[download_gptsovits_btn, download_status]
            )

        with gr.Column():
            gr.Markdown("### SoVITS Trained Models")
            download_sovits_btn = gr.Button("Download Trained SoVITS Models", variant="primary")
            sovits_status = gr.Textbox(label="Download Status", interactive=False)

            download_sovits_btn.click(
                fn=download_trained_models,
                outputs=[download_sovits_btn, sovits_status]
            )

    return download_gptsovits_btn, download_status


def ui_workbench():
    global all_gradio_components
    all_gradio_components = {}

    # Initialize the synthesizer and related components
    try:
        tts_synthesizer, tts_task_example = initialize_synthesizer()
    except Exception as e:
        gr.Markdown(f"Error initializing synthesizer: {e}")
        gr.Markdown("Please check if the model is installed and downloaded correctly.")
        print(f"Error initializing synthesizer: {e}")
        return

    # Get settings and configs
    ref_settings = tts_synthesizer.ui_config.get("ref_settings", [])
    basic_settings = tts_synthesizer.ui_config.get("basic_settings", [])
    advanced_settings = tts_synthesizer.ui_config.get("advanced_settings", [])
    params_config = tts_task_example.params_config
    has_character_param = True if "character" in params_config else False

    try:
        max_text_length = app_config.max_text_length
    except:
        max_text_length = -1

    from gpt_sovits.webuis.builders.gradio_builder import GradioTabBuilder

    with gr.Row():
        max_text_length_tip = (
            ""
            if max_text_length == -1
            else f"( Max allowed length: {max_text_length} )"
        )
        text = gr.Textbox(
            value="我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样。",
            label=f"Input Text {max_text_length_tip}",
            interactive=True,
            lines=8,
        )
        text.blur(
            lambda x: gr.update(
                value=cut_sentence_multilang(x, max_length=max_text_length)[0]
            ),
            [text],
            [text],
        )
        all_gradio_components["text"] = text
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label="Character Options", visible=has_character_param):
                    with gr.Group():
                        character, emotion, characters_and_emotions_ = (
                            change_character_list()
                        )
                        characters_and_emotions = gr.State(characters_and_emotions_)
                        scan_character_list = gr.Button(
                            "Scan Character List", variant="secondary"
                        )
                    all_gradio_components["character"] = character
                    all_gradio_components["emotion"] = emotion
                    character.change(
                        load_character_emotions,
                        inputs=[character, characters_and_emotions],
                        outputs=[emotion],
                    )

                    scan_character_list.click(
                        change_character_list,
                        inputs=[character, emotion],
                        outputs=[character, emotion, characters_and_emotions],
                    )
                with gr.Tab(label="Reference Settings"):
                    ref_settings_tab = GradioTabBuilder(ref_settings, params_config)
                    ref_settings_components = ref_settings_tab.build()
                    all_gradio_components.update(ref_settings_components)
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label="Basic Settings"):
                    basic_settings_tab = GradioTabBuilder(
                        basic_settings, params_config
                    )
                    basic_settings_components = basic_settings_tab.build()
                    all_gradio_components.update(basic_settings_components)
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label="Advanced Settings"):
                    advanced_settings_tab = GradioTabBuilder(
                        advanced_settings, params_config
                    )
                    advanced_settings_components = advanced_settings_tab.build()
                    all_gradio_components.update(advanced_settings_components)
    with gr.Tabs():
        with gr.Tab(label="Full Audio Request"):
            with gr.Row():
                get_full_audio_button = gr.Button(
                    "Generate Audio", variant="primary"
                )
                full_audio = gr.Audio(
                    None, label="Audio Output", type="filepath", streaming=False
                )
                get_full_audio_button.click(
                    lambda: gr.update(interactive=False),
                    None,
                    [get_full_audio_button],
                ).then(
                    get_audio,
                    inputs=[value for key, value in all_gradio_components.items()],
                    outputs=[full_audio],
                ).then(
                    lambda: gr.update(interactive=True),
                    None,
                    [get_full_audio_button],
                )
        with gr.Tab(label="Streaming Audio"):
            with gr.Row():
                get_streaming_audio_button = gr.Button(
                    "Generate Streaming Audio", variant="primary"
                )
                streaming_audio = gr.Audio(
                    None,
                    label="Audio Output",
                    type="filepath",
                    streaming=True,
                    autoplay=True,
                )
                get_streaming_audio_button.click(
                    lambda: gr.update(interactive=False),
                    None,
                    [get_streaming_audio_button],
                ).then(
                    get_streaming_audio,
                    inputs=[value for key, value in all_gradio_components.items()],
                    outputs=[streaming_audio],
                ).then(
                    lambda: gr.update(interactive=True),
                    None,
                    [get_streaming_audio_button],
                )

    gr.HTML("<hr style='border-top: 1px solid #ccc; margin: 20px 0;' />")
    gr.HTML(
        f"""<p>This is GPT-SoVITS TTS Extension. Current version: <a href="https://github.com/X-T-E-R/GPT-SoVITS-Inference">{frontend_version}</a> Project repository: <a href="https://github.com/X-T-E-R/GPT-SoVITS-Inference">Github</a></p>
            <p>For questions or more information, please refer to the documentation: <a href="https://github.com/rsxdalv/extension_gpt_sovits">Click to view detailed documentation</a></p>"""
    )


def ui_core():
    with gr.Tabs() as app:
        gr.Markdown("# GPT-SoVITS Text-to-Speech")

        with gr.Tab("Workbench"):
            ui_workbench()
        download_tab()

    return app


def ui_app():
    with gr.Blocks() as app:
        ui_core()
    return app
