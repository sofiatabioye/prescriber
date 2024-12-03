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
from load_queries import load_queries
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

custom_prompt_template = ChatPromptTemplate.from_template(
    "You are a highly knowledgeable medical assistant. Your goal is to provide detailed, evidence-based responses "
    "to queries related to SMPC, medical guidelines, and journal information.\n\n"
    "You have access to the following tools:\n"
    "{tools}\n\n"
    "1. SMPC Query Tool: For searching the medication summary of product characteristics for information about any medication.\n"
    "2. Guidance Query Tool: Should only be used when the query relates to Lynch syndrome for searching medical guidelines related to Lynch syndrome.\n"
    "3. Journal Query Tool: Should only be used when the query relates to Lynch syndrome for searching medical journals related to Lynch syndrome.\n\n"
    "When you are uncertain or need specific information, you **must** use these tools to find relevant details. Never provide a final answer without first attempting to use the appropriate tools.\n\n"
    "Follow these steps:\n"
    "- Always first review the previous conversation to understand if there is missing context, such as a medication name or topic.\n"
    "- If a medication name is mentioned in the current query or in the previous conversation, you **must** use the SMPC Query Tool to gather all relevant information about that medication.\n"
    "- If the SMPC Query Tool does not provide specific information about the medication, **explicitly mention this as part of your reasoning**, and proceed with any other relevant actions.\n"
    "- **Do not provide a Final Answer until all possible actions are taken.**\n"
    "- If the current query seems incomplete (e.g., missing medication name), try to infer it from the chat history.\n"
    "- If you cannot infer the missing details, ask the user to clarify before proceeding.\n"
    "- Think step by step to understand the question and decide on the next action.\n"
    "- If you determine that an action is needed, specify it clearly with an `Action:`.\n"
    "- Use the appropriate tools to gather information whenever it is needed. Do **not** skip the action step if a tool can be used to find the answer.\n"
    "- If an action fails to yield the required information, explain this in your **Thought** and reassess if another action is needed before providing a final answer.\n"
    "- Only provide a final answer after gathering all possible relevant information using the tools, or after exhausting all possible actions.\n\n"
    "Previous conversation:\n{chat_history}\n\n"
    "Current Query: {input}\n\n"
    "Use the following tools if needed:\n"
    "{tool_names}\n\n"
    "Begin your reasoning step-by-step and specify when you take an action:\n\n"
    "Thought: First, review the previous conversation to understand the context. What do you want to do first?\n"
    "{agent_scratchpad}\n\n"
    "If an action is needed, use the following format:\n"
    "Action: [Name of Tool]\nAction Input: [Input to the Tool]\n\n"
    "If an action fails or no specific information is found, include this fact in your reasoning and proceed accordingly.\n\n"
    "If you need more information from the user, ask them clearly.\n\n"
    "If no action is needed, and you have all the information required, provide the final answer in this format:\n"
    "Final Answer: [Your response here]"
)


# Load the REACT prompt
prompt = hub.pull("hwchase17/react")
print(prompt, "prompt")
# Create the REACT agent
agent = create_react_agent(tools=tools, llm=llm, prompt=custom_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)


# Streamlit UI
# Load sample queries from the text files
lynch_queries = load_queries("lynch_queries.txt")
general_queries = load_queries("smpc_queries.txt")

# Section for Lynch Syndrome Guidelines Related Questions
st.sidebar.subheader("Lynch Syndrome Patient Profiles for Prescribing")
selected_query = st.sidebar.radio("Choose a medication-related query:", general_queries, key="general_query_radio")

# Section for Medication Related Lynch Syndrome Cancer Patients
st.sidebar.subheader("Lynch Syndrome Guidelines Queries")
selected_med_query = st.sidebar.radio("Choose a guideline-related query:", lynch_queries, key="med_query_radio")


st.title("Medical Query Assistant")
# Initialize session state for chat history

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display the selected query from the sidebar in the text area
if selected_query and selected_query != "":
    st.session_state['query'] = selected_query
elif selected_med_query and selected_med_query != "":
    st.session_state['query'] = selected_med_query

query = st.text_area("Enter your query here:", value=st.session_state.get('query', ''), height=200)

# Clear the selection after query is displayed
if "query" in st.session_state:
    del st.session_state["query"]

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
