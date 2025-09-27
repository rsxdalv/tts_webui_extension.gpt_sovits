import setuptools

setuptools.setup(
    name="tts_webui_extension.gpt_sovits",
    packages=setuptools.find_namespace_packages(),
    version="0.1.1",
    author="rsxdalv",
    description="GPT-SoVITS: A TTS solution powered by GPT and SoftVC VITS Singing Voice Conversion.",
    url="https://github.com/rsxdalv/tts_webui_extension.gpt_sovits",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        "gpt_sovits @ https://github.com/rsxdalv/GPT-SoVITS/releases/download/v2.6.3/gpt_sovits-2.6.3-py3-none-any.whl",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)