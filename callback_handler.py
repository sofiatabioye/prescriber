from langchain.callbacks.base import BaseCallbackHandler
import logging

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []  # Store logs in a list

    def on_agent_action(self, action, **kwargs):
        # Log agent action
        log_entry = f"Agent Action: {action.tool} - Input: {action.tool_input}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_tool_start(self, tool, input_str, **kwargs):
        # Log when a tool is called
        log_entry = f"Tool Start: {tool.name} - Input: {input_str}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_tool_end(self, output, **kwargs):
        # Log the output of a tool
        log_entry = f"Tool End: Output: {output}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Log when the LLM starts processing
        log_entry = f"LLM Start: Prompts: {prompts}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_llm_end(self, response, **kwargs):
        # Log the response from the LLM
        log_entry = f"LLM End: Response: {response.generations}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_agent_finish(self, finish, **kwargs):
        # Log agent finishing
        log_entry = f"Agent Finished: {finish}"
        self.logs.append(log_entry)
        logging.info(log_entry)
    
    def on_error(self, error, **kwargs):
        # Capture and log the error
        log_entry = f"Error: {str(error)}"
        self.logs.append(log_entry)
        logging.error(log_entry)

    def get_logs(self):
        return self.logs
