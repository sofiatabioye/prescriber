import streamlit as st
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from callback_handler import StreamlitCallbackHandler
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
logging.basicConfig(level=logging.INFO)

# Initialize callback handler
callback_handler = StreamlitCallbackHandler()
# Set up the callback manager
callback_manager = CallbackManager(handlers=[callback_handler])

# Access the secrets
pinecone_api_key = st.secrets["default"]["PINECONE_API_KEY"]  # Securely manage keys
openai_api_key = st.secrets["default"]["OPENAI_API_KEY"]
langchain_api_key =  st.secrets["default"]["LANGCHAIN_API_KEY"]

# Pinecone setup
index_names = {"smpc": "smpc", "guidelines": "guidelines", "journals": "journals"}
pineconeIndex = "smpc"
pc = Pinecone(api_key=pinecone_api_key)

# Function to create query tools
def create_query_tool(index_name, tool_name, description):
    index = pc.Index(pineconeIndex)
    embedding_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = PineconeVectorStore(
        pinecone_index=index,
        embedding=embedding_model,
        namespace=index_name,
    )
    query_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding_model)

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

            # Extract response and metadata
            response_text = results.response if results.response else "No relevant information found."
            metadata_list = []
            for node in results.source_nodes:
                source_info = node.node.metadata.get("source", "Unknown Source")
                medication_name = node.node.metadata.get("medication_name", "Unknown Medication")
                section_title = node.node.metadata.get("section_title", "Unknown Section")
                title = node.node.metadata.get("title", "Unknown Title")
                metadata_list.append(f"Source: {source_info} - {medication_name} / {section_title} - {title}")
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

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
tools = [smpc_tool, guidance_tool, journal_tool]
# Prompt setup
prompt_template = ChatPromptTemplate.from_template(
    "You are a medical expert assistant using Pinecone to query multiple indexes. "
    "You can search the SMPC index, the Guidelines index, and the Journals index. "
    "Provide answers based on these indexes, including sources and metadata.\n\n"
    "Query: {input}\nAnswer:"
)
# Load the REACT prompt
prompt = hub.pull("hwchase17/react")

# Create the REACT agent
agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)
# Streamlit UI
st.title("Medical Query Assistant")
# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

query = st.text_area("Enter your query here:", height=200)
if st.button("Submit Query"):
    with st.spinner("Processing your query..."):
        try:
            # Prepare the chat history string
            chat_history = "\n".join(st.session_state['chat_history'])
            # Invoke the agent with the query
            response = agent_executor.invoke({
                "input": query,
                "chat_history": chat_history,
            }, {"callbacks": [callback_handler]})

           # Extract and display the final response
            final_response = response['output']
            st.success("Query Processed!")
            st.text_area("Response", final_response, height=300)

            # Update and store chat history
            st.session_state['chat_history'].append(f"Human: {query}")
            st.session_state['chat_history'].append(f"AI: {final_response}")

            # Display chat history
            st.subheader("Chat History")
            st.text_area("Chat History", chat_history + f"\nHuman: {query}\nAI: {final_response}", height=300)

            # Display logs
            logs = callback_handler.get_logs()
            if logs:
                log_content = "\n".join([f"<div>{log}</div>" for log in logs])
            else:
                log_content = "<div>No intermediate steps captured.</div>"
            st.subheader("Intermediate Steps")
            # Use custom HTML and CSS to make the container scrollable
            st.markdown(
                f"""
                <div style="height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
                    {log_content}
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as e:
            # Log and display errors
            st.error(f"An error occurred: {e}")
            logging.error(f"Error occurred during query processing: {e}")
            logs = callback_handler.get_logs()
            for log in logs:
                st.write(log)
