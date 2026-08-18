import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Document chunks
documents = [
    "Artificial Intelligence is the field of computer science that develops systems capable of performing tasks that normally require human intelligence.",

    "Machine Learning is a subset of Artificial Intelligence. It allows computers to learn patterns from data without being explicitly programmed.",

    "Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to process complex data.",

    "Natural Language Processing allows computers to understand and process human language."
]

# Generate embeddings
embeddings = model.encode(documents)

# Convert embeddings to NumPy float32
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to vector database
index.add(embeddings)

print("RAG Document Question Answering System")
print("Number of document chunks:", index.ntotal)

# Get question
question = input("\nEnter your question: ")

# Generate question embedding
question_embedding = model.encode([question])
question_embedding = np.array(question_embedding).astype("float32")

# Search for most relevant document
k = 1
distances, indices = index.search(question_embedding, k)

# Retrieve relevant document
answer = documents[indices[0][0]]

# Display result
print("\nQuestion:")
print(question)

print("\nRetrieved Context:")
print(answer)

print("\nAnswer:")
print(answer)

print("\nSimilarity Distance:")
print(distances[0][0])