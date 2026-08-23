import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Engineering Machine Translation",
    page_icon="🌐",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🌐 Engineering Machine Translation")

st.write(
    "Translate engineering-related English text "
    "into Indian languages using a pretrained AI model."
)


# =====================================================
# LANGUAGE OPTIONS
# =====================================================

languages = {
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva"
}


# =====================================================
# SELECT LANGUAGE
# =====================================================

language = st.selectbox(
    "Select target Indian language:",
    list(languages.keys())
)


# =====================================================
# TEXT INPUT
# =====================================================

text = st.text_area(
    "Enter English engineering text:",
    placeholder=(
        "Example: Artificial intelligence is "
        "used in modern engineering applications."
    ),
    height=180
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_name = (
        "ai4bharat/indictrans2-en-indic-dist-200M"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    return tokenizer, model


# =====================================================
# TRANSLATE
# =====================================================

if st.button("🌐 Translate"):

    if text.strip() == "":

        st.warning(
            "Please enter some English text."
        )

    else:

        with st.spinner(
            f"Translating into {language}..."
        ):

            try:

                tokenizer, model = load_model()

                target_language = languages[
                    language
                ]


                # IndicTrans2 format

                source_text = (
                    text.strip()
                    + " "
                    + target_language
                )


                # Tokenize

                inputs = tokenizer(
                    source_text,
                    return_tensors="pt"
                )


                # Generate translation

                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5
                )


                # Decode

                translated_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )


                st.success(
                    "Translation completed!"
                )


                st.subheader(
                    f"🇮🇳 {language} Translation"
                )

                st.write(
                    translated_text
                )


                st.download_button(
                    label="⬇️ Download Translation",
                    data=translated_text,
                    file_name="translated_engineering_text.txt",
                    mime="text/plain"
                )


            except Exception as e:

                st.error(
                    "Translation failed."
                )

                st.code(
                    str(e)
                )


# =====================================================
# EXAMPLES
# =====================================================

st.divider()

st.subheader(
    "💡 Example Engineering Text"
)

st.write(
    "Artificial intelligence is used in modern "
    "engineering applications to solve complex "
    "problems."
)

st.write(
    "A microcontroller is an integrated circuit "
    "used to control electronic devices."
)

st.write(
    "Renewable energy technologies help reduce "
    "environmental pollution."
)