import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a branch of artificial intelligence.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Java is an object-oriented programming language.",
    "Deep learning uses neural networks for complex tasks."
]

# Generate embeddings
embeddings = model.encode(documents)

# Convert embeddings to float32
embeddings = np.array(embeddings).astype("float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)

print("Vector database created successfully!")
print("Number of documents:", index.ntotal)

# Get query from user
query = input("\nEnter your query: ")

# Generate query embedding
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Number of results to retrieve
k = 2

# Search FAISS database
distances, indices = index.search(query_embedding, k)

# Display results
print("\nMost Similar Documents:")

for i in range(k):
    print("\nDocument:", documents[indices[0][i]])
    print("Distance:", distances[0][i])