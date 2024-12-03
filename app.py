import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)

# Pinecone and OpenAI API keys
pinecone_api_key = st.secrets["PINECONE_API_KEY"]  # Securely manage keys
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Pinecone configuration
index_names = {"smpc": "smpc", "guidelines": "guidelines", "journals": "journals"}
pineconeIndex = "smpc"
pc = Pinecone(api_key=pinecone_api_key)

def create_query_tool(index_name, tool_name, description):
    index = pc.Index(pineconeIndex)
    embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
    vector_store = PineconeVectorStore(
        pinecone_index=index,
        embedding=embedding_model,
        namespace=index_name,
    )
    query_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    class QueryInput(BaseModel):
        query: str = Field(..., description="The query to search in the index")

    def query_function(input_data=None, **kwargs):
        try:
            if input_data:
                if isinstance(input_data, dict):
                    input_data = QueryInput(**input_data)
                elif isinstance(input_data, str):
                    input_data = QueryInput(query=input_data)
            else:
                input_data = QueryInput(**kwargs)

            query_engine = query_index.as_query_engine(similarity_top_k=3)
            results = query_engine.query(input_data.query)
            response_text = results.response if results.response else "No relevant information found."
            metadata_list = [
                f"Source: {node.node.metadata.get('source', 'Unknown Source')}"
                for node in results.source_nodes
            ]
            formatted_metadata = "\n".join(metadata_list)
            return f"Response: {response_text}\n\nMetadata:\n{formatted_metadata}"
        except Exception as e:
            return f"Error querying the index '{index_name}': {str(e)}"

    return Tool(
        name=tool_name,
        func=query_function,
        description=description,
    )

# Create tools
smpc_tool = create_query_tool("smpc", "SmpcQuery", "Search SMPC Index.")
guidance_tool = create_query_tool("guidelines", "GuidanceQuery", "Search Guidance Index.")
journal_tool = create_query_tool("journals", "JournalQuery", "Search Journals Index.")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [smpc_tool, guidance_tool, journal_tool]
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# Streamlit UI
st.title("Medical Query Assistant")
query = st.text_area("Enter your query here:", height=200)
if st.button("Submit Query"):
    with st.spinner("Processing your query..."):
        try:
            response = agent.run(query)
            st.success("Query Processed!")
            st.text_area("Response", response, height=300)
        except Exception as e:
            st.error(f"An error occurred: {e}")
