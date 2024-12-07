from langchain.callbacks.base import BaseCallbackHandler
import logging

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, response_placeholder=None):
        self.logs = []
        self.response_placeholder = response_placeholder

    def on_agent_action(self, action, **kwargs):
        # When the agent decides to use a tool, let the user know
        if self.response_placeholder:
            if action.tool == "SmpcQuery":
                self.response_placeholder.markdown("**Please wait, fetching data from SMPC...**")
            elif action.tool == "GuidanceQuery":
                self.response_placeholder.markdown("**Please wait, checking the guidelines...**")
            elif action.tool == "JournalQuery":
                self.response_placeholder.markdown("**Please wait, checking the journals...**")
            else:
                self.response_placeholder.markdown("**Please wait, processing your request...**")

        log_entry = f"Agent Action: {action.tool} - Input: {action.tool_input}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_tool_start(self, tool, input_str, **kwargs):
        # When a tool starts, we can also show a waiting message
        tool_name = tool['name']
        if self.response_placeholder:
            if tool_name == "SmpcQuery":
                self.response_placeholder.markdown("**Please wait, fetching data from SMPC...**")
            elif tool_name == "GuidanceQuery":
                self.response_placeholder.markdown("**Please wait, checking the guidelines...**")
            elif tool_name == "JournalQuery":
                self.response_placeholder.markdown("**Please wait, checking the journals...**")
            else:
                self.response_placeholder.markdown("**Please wait, processing your request...**")

        log_entry = f"Tool Start: {tool['name']} - Input: {input_str}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_tool_end(self, output, **kwargs):
        log_entry = f"Tool End: Output: {output}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_llm_start(self, serialized, prompts, **kwargs):
        # When the LLM starts generating a response
        if self.response_placeholder:
            self.response_placeholder.markdown("**Please wait, preparing your answer...**")

        log_entry = f"LLM Start: Prompts: {prompts}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_llm_new_token(self, token: str, **kwargs):
        # If you want to show partial tokens, handle them here
        # Right now, we do nothing since we want a static message
        pass

    def on_llm_end(self, response, **kwargs):
        log_entry = f"LLM End: Response: {response.generations}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_agent_finish(self, finish, **kwargs):
        log_entry = f"Agent Finished: {finish}"
        self.logs.append(log_entry)
        logging.info(log_entry)

    def on_error(self, error, **kwargs):
        log_entry = f"Error: {str(error)}"
        self.logs.append(log_entry)
        logging.error(log_entry)

    def get_logs(self):
        return self.logs
