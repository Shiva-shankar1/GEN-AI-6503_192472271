import streamlit as st
from transformers import pipeline
from pypdf import PdfReader


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Engineering Document Summarizer",
    page_icon="📄",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("📄 Engineering Document Summarizer")

st.write(
    "Upload an engineering PDF and generate "
    "a short meaningful summary using AI."
)


# =====================================================
# LOAD AI MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    return model


# =====================================================
# EXTRACT TEXT FROM PDF
# =====================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# =====================================================
# SPLIT TEXT
# =====================================================

def split_text(text, words_per_chunk=250):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        words_per_chunk
    ):

        chunk = " ".join(
            words[i:i + words_per_chunk]
        )

        chunks.append(chunk)

    return chunks


# =====================================================
# UPLOAD PDF
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload Engineering PDF",
    type=["pdf"]
)


# =====================================================
# PROCESS PDF
# =====================================================

if uploaded_file is not None:

    st.success(
        "PDF uploaded successfully!"
    )


    if st.button(
        "📝 Generate Summary"
    ):

        with st.spinner(
            "Reading document..."
        ):

            try:

                # -------------------------------------
                # Extract text
                # -------------------------------------

                text = extract_pdf_text(
                    uploaded_file
                )


                if not text.strip():

                    st.error(
                        "No readable text found in PDF."
                    )

                    st.stop()


                # -------------------------------------
                # Document information
                # -------------------------------------

                st.subheader(
                    "📊 Document Information"
                )

                st.write(
                    "Characters:",
                    len(text)
                )

                st.write(
                    "Words:",
                    len(text.split())
                )


                # -------------------------------------
                # Create chunks
                # -------------------------------------

                chunks = split_text(
                    text
                )


                # -------------------------------------
                # Load model
                # -------------------------------------

                generator = load_model()


                summaries = []


                # -------------------------------------
                # Generate summaries
                # -------------------------------------

                for number, chunk in enumerate(
                    chunks[:5]
                ):

                    st.write(
                        f"Processing section "
                        f"{number + 1}..."
                    )


                    prompt = (
                        "Summarize the following "
                        "engineering document section "
                        "in a short and meaningful way:\n\n"
                        + chunk
                        + "\n\nSummary:"
                    )


                    result = generator(
                        prompt,
                        max_new_tokens=80,
                        do_sample=False
                    )


                    generated_text = result[0][
                        "generated_text"
                    ]


                    # Remove prompt from output

                    if "Summary:" in generated_text:

                        summary = generated_text.split(
                            "Summary:",
                            1
                        )[1].strip()

                    else:

                        summary = generated_text


                    summaries.append(
                        summary
                    )


                # -------------------------------------
                # Combine summaries
                # -------------------------------------

                final_summary = "\n\n".join(
                    summaries
                )


                # -------------------------------------
                # Display summary
                # -------------------------------------

                st.subheader(
                    "📋 AI Generated Summary"
                )

                st.write(
                    final_summary
                )


                # -------------------------------------
                # Download
                # -------------------------------------

                st.download_button(
                    label="⬇️ Download Summary",
                    data=final_summary,
                    file_name="engineering_summary.txt",
                    mime="text/plain"
                )


            except Exception as e:

                st.error(
                    "Summarization failed."
                )

                st.code(
                    str(e)
                )


# =====================================================
# EXAMPLES
# =====================================================

st.divider()

st.subheader(
    "💡 Suitable Engineering Documents"
)

st.write(
    "• Artificial Intelligence notes"
)

st.write(
    "• Machine Learning research paper"
)

st.write(
    "• Embedded Systems document"
)

st.write(
    "• Electrical Engineering notes"
)

st.write(
    "• Mechanical Engineering report"
)