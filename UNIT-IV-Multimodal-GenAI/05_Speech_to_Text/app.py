import streamlit as st
from transformers import pipeline


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Engineering Speech to Text",
    page_icon="🎤",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🎤 Engineering Speech-to-Text")

st.write(
    "Speak an engineering-related question "
    "and convert your speech into written text using AI."
)


# =====================================================
# LOAD WHISPER MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny"
    )

    return model


# =====================================================
# AUDIO INPUT
# =====================================================

audio = st.audio_input(
    "🎤 Click here and record your engineering question"
)


# =====================================================
# PROCESS AUDIO
# =====================================================

if audio is not None:

    st.subheader("🔊 Recorded Audio")

    st.audio(
        audio
    )


    if st.button("📝 Convert Speech to Text"):

        with st.spinner(
            "Converting speech to text..."
        ):

            try:

                # Load Whisper model
                model = load_model()


                # Convert speech to text
                result = model(
                    audio.getvalue(),
                    generate_kwargs={
                        "language": "english",
                        "task": "transcribe"
                    }
                )


                # Get recognized text
                text = result["text"].strip()


                # Display success
                st.success(
                    "Speech converted successfully!"
                )


                # Display result
                st.subheader(
                    "📝 Recognized Engineering Query"
                )

                st.write(
                    text
                )


                # Allow copying/viewing the text
                st.text_area(
                    "Recognized Text",
                    value=text,
                    height=100
                )


            except Exception as e:

                st.error(
                    "Speech recognition failed."
                )

                st.code(
                    str(e)
                )


# =====================================================
# EXAMPLE QUESTIONS
# =====================================================

st.subheader(
    "💡 Example Engineering Questions"
)

st.write(
    "🎤 What is Ohm's Law?"
)

st.write(
    "🎤 Explain the working of a transformer."
)

st.write(
    "🎤 What is a microcontroller?"
)

st.write(
    "🎤 Explain the PID controller."
)

st.write(
    "🎤 What is machine learning?"
)

st.write(
    "🎤 What is the difference between a "
    "microprocessor and a microcontroller?"
)


# =====================================================
# INFORMATION
# =====================================================

st.divider()

st.info(
    "Tip: Speak clearly in English and keep the "
    "microphone close to you for better recognition."
)