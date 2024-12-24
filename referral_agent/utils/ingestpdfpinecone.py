import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Set your environment variables

# Initialize Pinecone
index_name = "pathways"
index = pc.Index(index_name)

# Load the PDF file
loader = PyPDFLoader("./rules/BSOL_Pathway.pdf")
documents = loader.load()

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Convert texts to embeddings 
embeddings_list = []
for text in texts:
    try:
        embedding = embeddings.embed_query(text.page_content)
        embeddings_list.append(embedding)
    except Exception as e:
        print(f"Error embedding text: {e}")
        continue

vectors = [
    {
        "id": f"doc-{i}",
        "values": embedding,
        "metadata": {"text": text.page_content, "document": "BSOL_Pathway.pdf", "source": "Bsol"}
    }
    for i, (text, embedding) in enumerate(zip(texts, embeddings_list))
]

index.upsert(vectors=vectors, namespace="ns1")

print(f"Successfully ingested {len(texts)} documents into Pinecone.")
