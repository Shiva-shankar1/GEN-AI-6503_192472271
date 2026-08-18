import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------------
# 1. Read external document
# -----------------------------------

with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()


# -----------------------------------
# 2. Split document into chunks
# -----------------------------------

documents = [
    line.strip()
    for line in text.split("\n")
    if line.strip()
]


# -----------------------------------
# 3. Load embedding model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 4. Generate document embeddings
# -----------------------------------

embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")


# -----------------------------------
# 5. Create FAISS vector database
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print("AI Assistant Started")
print("External document loaded successfully!")
print("Number of document sections:", index.ntotal)


# -----------------------------------
# 6. Ask questions
# -----------------------------------

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Assistant: Goodbye!")
        break


    # Generate query embedding
    query_embedding = model.encode([question])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # Search vector database
    k = 1

    distances, indices = index.search(
        query_embedding,
        k
    )


    # Retrieve relevant document
    answer = documents[indices[0][0]]


    # Display answer
    print("\nAssistant:", answer)