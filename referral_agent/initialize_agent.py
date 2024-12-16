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
        "You analyze 2WW referral forms to provide evidence-based recommendations for the next steps, based on established guidelines."
        "You have access to the following tools:\n"
        "{tools}\n\n"
        "GuidelineQuery: Use this tool to query colorectal cancer guideline documents and determine the appropriate next action."
        "Your tasks:"
        "1. Carefully analyze the extracted patient data."
        "2. Use the GuidanceQuery tool to search for guideline-based recommendations."
        "3. Provide a clear and concise recommendation for the next steps."
        "Input Query: {input}"
        "Begin your reasoning step-by-step:"
        "Thought: What is the first step based on the data provided?"
        "{agent_scratchpad}\n\n"
        "If you take an action, use the following format:"
        "Action: [Name of Tool]"
        "Action Input: [Input to the Tool]"
        "If no further action is needed, provide your final recommendation in this format:"
        "Final Answer:"
        "Patient Name: [Patient name in the referral form]"
        "Hospital Number: [The patient's hospital/NHS number in the referral form]"
        "Outcome: [The recommended next action]"
        "Referral: [The referral made e.g. LGI 2WW REFERRAL MADE, Advice &​Guidance/​Routine referral, 2WW NSS ​Referral, GP Direct access urgent colonoscopy​ and others ]"
        "Rationale: [The guideline-based reasoning for the outcome]"
        "- If you cannot infer the missing details, ask the user to clarify before proceeding."
        "- Think step by step to understand the question and decide on the next action."
        "- If you determine that an action is needed, specify it clearly with an `Action:`."
        "- Use the appropriate tools to gather information whenever it is needed. Do **not** skip the action step if a tool can be used to find the answer."
        "- If an action fails to yield the required information, explain this in your **Thought** and reassess if another action is needed before providing a final answer."
        "- Only provide a final answer after gathering all possible relevant information using the tools, or after exhausting all possible actions."
        "Previous conversation: {chat_history} \n\n"
        "Current Query: {input} \n\n"
        "Use the following tools if needed:\n"
        "{tool_names}\n\n"
        "Begin your reasoning step-by-step and specify when you take an action:"
        "Thought: First, review the previous conversation to understand the context. What do you want to do first?"
        "{agent_scratchpad} \n\n"
        "If an action is needed, use the following format:"
        "Action: [Name of Tool] Action Input: [Input to the Tool]"
        "If an action fails or no specific information is found, include this fact in your reasoning and proceed accordingly."
        "If you need more information from the user, ask them clearly."
        "If no action is needed, and you have all the information required, provide the final answer in this format:"
        "Final Answer: [Your response here]"
    )


    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1500
    )

    agent = create_react_agent(tools=tools, llm=llm, prompt=prompt_template)

    return AgentExecutor(agent=agent, tools=[guideline_tool], handle_parsing_errors=True, verbose=True)
