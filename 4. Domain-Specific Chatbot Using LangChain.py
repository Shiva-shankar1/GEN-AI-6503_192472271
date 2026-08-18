from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------
# 1. Domain-specific information
# --------------------------------

text = """
Artificial Intelligence is a field of computer science that creates
systems capable of performing tasks that normally require human intelligence.

Machine Learning is a subset of Artificial Intelligence. It allows computers
to learn patterns from data without being explicitly programmed.

Deep Learning is a subset of Machine Learning that uses neural networks
with multiple layers to solve complex problems.

Natural Language Processing is a branch of Artificial Intelligence that
allows computers to understand and process human language.

Computer Vision is a field of Artificial Intelligence that enables computers
to understand and analyze images and videos.

FAISS is a library used for efficient similarity search and clustering
of dense vectors.

Generative AI is a type of Artificial Intelligence that can generate new
content such as text, images, audio, and code.
"""


# --------------------------------
# 2. Split the text into chunks
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

documents = splitter.create_documents([text])


# --------------------------------
# 3. Create embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# --------------------------------
# 4. Create FAISS vector database
# --------------------------------

vector_db = FAISS.from_documents(
    documents,
    embeddings
)

print("Domain-specific chatbot created successfully!")


# --------------------------------
# 5. Create retriever
# --------------------------------

retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------
# 6. Chatbot
# --------------------------------

print("\nAI Chatbot")
print("Ask questions about Artificial Intelligence.")
print("Type 'exit' to stop.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Bot: Goodbye!")
        break

    # Retrieve relevant documents
    results = retriever.invoke(question)

    print("\nBot:")

    for result in results:
        print(result.page_content)