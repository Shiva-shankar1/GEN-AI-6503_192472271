import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("📄 AI Resume Screening System")

st.write(
    "Upload engineering resumes and rank candidates "
    "according to a job description using AI."
)


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# =====================================================
# EXTRACT PDF TEXT
# =====================================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# =====================================================
# CALCULATE COSINE SIMILARITY
# =====================================================

def cosine_similarity(a, b):

    a = np.array(a)

    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# =====================================================
# JOB DESCRIPTION
# =====================================================

st.subheader("💼 Engineering Job Description")

job_description = st.text_area(
    "Enter the job description:",
    height=180,
    placeholder=(
        "Example: We are looking for a software engineer "
        "with Java, Python, SQL, machine learning, "
        "and problem-solving skills."
    )
)


# =====================================================
# RESUME UPLOAD
# =====================================================

st.subheader("📄 Upload Candidate Resumes")

resumes = st.file_uploader(
    "Upload resumes",
    type=["pdf"],
    accept_multiple_files=True
)


# =====================================================
# SCREEN RESUMES
# =====================================================

if st.button("🔍 Screen Resumes"):

    if not job_description.strip():

        st.warning(
            "Please enter a job description."
        )

    elif not resumes:

        st.warning(
            "Please upload at least one resume."
        )

    else:

        with st.spinner(
            "Analyzing resumes..."
        ):

            try:

                model = load_model()


                # -------------------------------------
                # Job description embedding
                # -------------------------------------

                job_embedding = model.encode(
                    job_description
                )


                results = []


                # -------------------------------------
                # Process resumes
                # -------------------------------------

                for resume in resumes:

                    resume_text = extract_pdf_text(
                        resume
                    )


                    if not resume_text.strip():

                        continue


                    resume_embedding = model.encode(
                        resume_text
                    )


                    similarity = cosine_similarity(
                        job_embedding,
                        resume_embedding
                    )


                    score = similarity * 100


                    results.append(
                        {
                            "Candidate": resume.name,
                            "Match Score": round(
                                score,
                                2
                            )
                        }
                    )


                # -------------------------------------
                # Sort candidates
                # -------------------------------------

                results.sort(
                    key=lambda x: x["Match Score"],
                    reverse=True
                )


                # -------------------------------------
                # Display ranking
                # -------------------------------------

                st.success(
                    "Resume screening completed!"
                )


                st.subheader(
                    "🏆 Candidate Ranking"
                )


                for rank, candidate in enumerate(
                    results,
                    start=1
                ):

                    st.write(
                        f"### {rank}. "
                        f"{candidate['Candidate']}"
                    )

                    st.progress(
                        min(
                            candidate["Match Score"] / 100,
                            1.0
                        )
                    )

                    st.write(
                        f"Match Score: "
                        f"**{candidate['Match Score']}%**"
                    )


                # -------------------------------------
                # Summary table
                # -------------------------------------

                st.subheader(
                    "📊 Screening Summary"
                )

                st.table(results)


            except Exception as e:

                st.error(
                    "Resume screening failed."
                )

                st.code(
                    str(e)
                )


# =====================================================
# INFORMATION
# =====================================================

st.divider()

st.info(
    "The system uses semantic similarity between "
    "the job description and each resume. A higher "
    "score indicates a stronger semantic match."
)