from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
documents = [
    "Python is a programming language.",
    "Java is used for software development.",
    "Machine learning is a branch of artificial intelligence.",
    "Football is a popular sport.",
    "Deep learning uses neural networks."
]

# Query
query = "What is artificial intelligence and machine learning?"

# Generate embeddings
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Calculate similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Find most similar document
best_index = similarities.argmax()

print("Query:", query)
print("\nMost Similar Document:")
print(documents[best_index])
print("Similarity Score:", similarities[best_index])