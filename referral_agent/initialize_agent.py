### Module 1: Query Tool and Agent Initialization ###

from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

def create_query_tool(index_name, tool_name, description, namespace="bsol"):
    """
    Creates a tool to query a specified Pinecone index.

    Args:
        index_name (str): Name of the Pinecone index.
        tool_name (str): Name of the tool.
        description (str): Description of the tool.
        namespace (str): Namespace for Pinecone index.

    Returns:
        Tool: A LangChain tool for querying the specified index.
    """
    pinecone_index = Pinecone.from_existing_index(index_name, OpenAIEmbeddings(), namespace=namespace)
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=pinecone_index.as_retriever()
    )
    return Tool(
        name=tool_name,
        func=retrieval_chain.run,
        description=description
    )

def initialize_agent():
    """Initializes the LangChain agent with the necessary tools and prompt."""
    guideline_tool = create_query_tool("pathways", "GuidelineQuery", "Search Guidance Index.")

    tools = [guideline_tool]
    tool_names = ["GuidelineQuery"];

    prompt_template = ChatPromptTemplate.from_template(
        "You are a highly knowledgeable medical assistant specializing in colorectal cancer referrals. "
        "You analyze 2WW referral forms to provide evidence-based recommendations for the next steps, based on colorectal cancer guidelines indexed in the GuidanceQuery tool."
        "\n\n"
        "You have access to the following tools:\n"
        "{tools}\n\n"
        "{tool_names}: Use this tool to query colorectal cancer guideline documents in the system and determine the appropriate next steps for the referral."
        "\n\n"
        "Your tasks:"
        "1. Carefully analyze the extracted patient data including age, gender, FIT result, symptoms, FIT positive and negative pathway results, WHO performance status, and additional history."
        "2. Use the GuidanceQuery tool to search the indexed guideline documents for appropriate recommendations."
        "3. Provide a recommendation based exclusively on the information retrieved from the guidelines in the system."
        "\n\n"
        "Do not use external knowledge like NICE guidelines; all recommendations must be derived from the GuidanceQuery tool."
        "\n\n"
        "Input Query: {input}"
        "\n\n"
        "Begin your reasoning step-by-step:"
        "Thought: Analyze the provided patient data and identify the key factors relevant for a recommendation."
        "{agent_scratchpad}\n\n"
        "If an action is needed, use the following format:"
        "Action: [Name of Tool]"
        "Action Input: [Input to the Tool]"
        "\n\n"
        "If an action fails or no specific information is found, explain this in your **Thought** and reassess whether another action is needed before providing a final answer."
        "\n\n"
        "Final recommendations must be provided in this format:"
        "Final Answer:"
        "Patient Name: [Extracted Patient Name]"
        "Hospital Number: [Extracted Hospital/NHS Number]"
        "Outcome: [Recommended action]"
        "Referral: [Referral type, e.g., LGI 2WW REFERRAL MADE, Advice & Guidance, Routine referral, or other referral types]"
        "Rationale: [Explain the guideline-based reasoning for the outcome]"
    )


    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1500
    )

    agent = create_react_agent(tools=tools, llm=llm, prompt=prompt_template)

    return AgentExecutor(agent=agent, tools=[guideline_tool], handle_parsing_errors=True, verbose=True)
