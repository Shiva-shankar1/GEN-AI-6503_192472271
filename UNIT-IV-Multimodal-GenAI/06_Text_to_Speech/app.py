import streamlit as st
import torch
import soundfile as sf

from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Engineering Text to Speech",
    page_icon="🔊",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🔊 Engineering Text-to-Speech")

st.write(
    "Enter engineering-related text and convert "
    "it into natural-sounding speech using AI."
)


# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts"
    )

    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts"
    )

    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan"
    )

    return processor, model, vocoder


# =====================================================
# TEXT INPUT
# =====================================================

text = st.text_area(
    "Enter engineering text:",
    placeholder=(
        "Example: A transformer is an electrical "
        "device that transfers electrical energy "
        "between circuits."
    ),
    height=150
)


# =====================================================
# EXAMPLE TEXT
# =====================================================

st.subheader("💡 Example Engineering Text")

st.write(
    "A transformer is an electrical device that "
    "transfers electrical energy from one circuit "
    "to another through electromagnetic induction."
)

st.write(
    "A microcontroller is a compact integrated "
    "circuit containing a processor, memory, and "
    "input and output peripherals."
)

st.write(
    "Artificial intelligence enables computers "
    "to perform tasks that normally require "
    "human intelligence."
)


# =====================================================
# GENERATE SPEECH
# =====================================================

if st.button("🔊 Generate Speech"):

    if text.strip() == "":

        st.warning(
            "Please enter some engineering text."
        )

    else:

        with st.spinner(
            "Generating speech..."
        ):

            try:

                # Load models

                processor, model, vocoder = (
                    load_models()
                )


                # Prepare text

                inputs = processor(
                    text=text,
                    return_tensors="pt"
                )


                # Create speaker embedding

                speaker_embeddings = torch.zeros(
                    (1, 512)
                )


                # Generate speech

                speech = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=speaker_embeddings,
                    vocoder=vocoder
                )


                # Save audio

                output_file = "engineering_speech.wav"

                sf.write(
                    output_file,
                    speech.numpy(),
                    16000
                )


                st.success(
                    "Speech generated successfully!"
                )


                # Play audio

                st.audio(
                    output_file,
                    format="audio/wav"
                )


                # Download audio

                with open(
                    output_file,
                    "rb"
                ) as audio_file:

                    st.download_button(
                        label="⬇️ Download Audio",
                        data=audio_file,
                        file_name="engineering_speech.wav",
                        mime="audio/wav"
                    )


            except Exception as e:

                st.error(
                    "Text-to-speech generation failed."
                )

                st.code(
                    str(e)
                )


# =====================================================
# INFORMATION
# =====================================================

st.divider()

st.info(
    "This application uses the pre-trained "
    "Microsoft SpeechT5 text-to-speech model."
)