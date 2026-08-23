import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(
    page_title="Engineering College AI Chatbot",
    page_icon="🎓"
)

st.title("🎓 Engineering College AI Chatbot")
st.write("Ask questions about the engineering college.")

# --------------------------------------------------
# 1. Load knowledge
# --------------------------------------------------

with open("knowledge.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

# --------------------------------------------------
# 2. Split knowledge into chunks
# --------------------------------------------------

chunks = [
    chunk.strip()
    for chunk in knowledge.split("\n\n")
    if chunk.strip()
]

# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

# --------------------------------------------------
# 5. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# --------------------------------------------------
# 6. Ask question
# --------------------------------------------------

question = st.text_input("Enter your question:")

if question:

    # Convert question into embedding
    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        k=3
    )

    # Get relevant information
    relevant_chunks = []

    for i in indices[0]:
        if i < len(chunks):
            relevant_chunks.append(chunks[i])

    context = "\n\n".join(relevant_chunks)

    # Display retrieved information
    st.subheader("Relevant Information")
    st.write(context)

    # Simple answer generation
    st.subheader("Answer")

    st.write(
        "Based on the college information provided:\n\n"
        + context
    )