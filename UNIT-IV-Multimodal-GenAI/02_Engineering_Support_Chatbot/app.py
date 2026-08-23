import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Engineering Support Chatbot",
    page_icon="🔧",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🔧 Engineering Support Chatbot")

st.write(
    "Ask technical engineering questions "
    "and receive AI-generated answers."
)


# =====================================================
# LOAD KNOWLEDGE BASE
# =====================================================

@st.cache_data
def load_knowledge():

    with open(
        "knowledge.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


chunks = load_knowledge()


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =====================================================
# CREATE FAISS INDEX
# =====================================================

@st.cache_resource
def create_faiss_index(_model, chunks):

    embeddings = _model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


index = create_faiss_index(
    embedding_model,
    chunks
)


# =====================================================
# LOAD LANGUAGE MODEL
# =====================================================

@st.cache_resource
def load_language_model():

    return pipeline(
        "text-generation",
        model="google/flan-t5-small"
    )


generator = load_language_model()


# =====================================================
# USER QUESTION
# =====================================================

question = st.text_input(
    "Enter your engineering question:",
    placeholder="Example: What is Ohm's Law?"
)


# =====================================================
# PROCESS QUESTION
# =====================================================

if question:

    # -------------------------------------------------
    # Create embedding for question
    # -------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )


    # -------------------------------------------------
    # Search FAISS
    # -------------------------------------------------

    distances, indices = index.search(
        question_embedding,
        k=min(3, len(chunks))
    )


    # -------------------------------------------------
    # Retrieve relevant information
    # -------------------------------------------------

    relevant_chunks = []

    for i in indices[0]:

        if i < len(chunks):

            relevant_chunks.append(
                chunks[i]
            )


    context = "\n\n".join(
        relevant_chunks
    )


    # =================================================
    # CREATE PROMPT
    # =================================================

    prompt = f"""
You are an engineering support assistant.

Answer the student's question using the context below.

Give a simple and clear explanation.

Context:
{context}

Question:
{question}

Answer:
"""


    # =================================================
    # GENERATE ANSWER
    # =================================================

    with st.spinner(
        "Generating AI answer..."
    ):

        result = generator(
            prompt,
            max_new_tokens=150,
            do_sample=False
        )


    answer = result[0]["generated_text"]


    # =================================================
    # DISPLAY ANSWER
    # =================================================

    st.subheader("🤖 AI Answer")

    st.write(answer)


    # =================================================
    # DISPLAY RETRIEVED INFORMATION
    # =================================================

    with st.expander(
        "View Retrieved Engineering Information"
    ):

        st.write(context)