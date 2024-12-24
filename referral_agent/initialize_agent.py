### Module 1: Query Tool and Agent Initialization ###

from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

def create_query_tool(index_name, tool_name, description, namespace="ns1"):
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
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=pinecone_index.as_retriever()
    )
    return Tool(
        name=tool_name,
        func=retrieval_chain.run,
        description=description
    )

def initialize_agent():
    """Initializes the LangChain agent with the necessary tools and prompt."""
    guideline_tool = create_query_tool("pathways", "GuidelineQuery", "Search the colorectal cancer guideline index for next steps.")

    tools = [guideline_tool]
    tool_names = ["GuidelineQuery"]

    template = '''You are a highly knowledgeable medical assistant specializing in colorectal cancer referrals. You analyze 2WW referral forms data to provide evidence-based recommendations for the next steps based on colorectal cancer guidelines.
        Answer the following questions as best you can. Imagine you are the GP and this is all you have about this patient. You have access to the following tools:
        {tools}
        Your tasks:
            1. Carefully analyze the extracted patient data (e.g., age, gender, FIT result, symptoms, WHO performance status, additional history).
            2. Use the tool to search the guideline documents for the appropriate recommended action and outcome.
            3. Provide a recommended action exclusively based on the information retrieved from those guidelines.
        Do not use external knowledge like NICE guidelines; all recommendations must be derived from the provided tools.

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of the tools in [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        INTERNAL REASONING GUIDANCE (Do NOT reveal this to the user):
        Analyze the input and reason carefully using 'Thought:'
        Always follow 'Thought:' with 'Action:' if additional data or tool inputs are needed.
        Capture tool outputs under 'Observation:' and iterate logically if necessary.
        Transition to 'Final Answer' as soon as sufficient information is gathered.
        Avoid redundant iterations and do not re-enter 'Thought:' once a valid final answer is ready

       STOPPING CONDITIONS:
        • Do NOT re-enter 'Thought → Action → Observation' once the final answer is generated.


        Begin!

        Question: Given the {input}, provide a recommended course of action and potential outcome
        Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1500
    )

    agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)

    return AgentExecutor(agent=agent, tools=[guideline_tool], handle_parsing_errors=True, verbose=True)