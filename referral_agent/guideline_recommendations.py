### Module 3: Recommendation Generation ###
import io
import sys
import re

def strip_ansi_codes(text):
    """Removes ANSI escape codes from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def get_guideline_recommendations(summary, agent_executor):
    """Generates recommendations and captures intermediate steps."""
    
    # Prepare the prompt
    prompt = f"""
        Based on the following patient data extracted from a patient's 2WW referral form:
        {summary}

        Please provide guideline-based recommendations for the patient. Use the medical guidelines tool to inform your response.
    """

    # Capture verbose output
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        # Execute the agent and capture logs
        response = agent_executor.invoke({"input": prompt, "chat_history": ""})
    finally:
        # Restore standard output
        sys.stdout = old_stdout

    # Retrieve verbose logs
    verbose_logs = mystdout.getvalue()

    # Extract final output and intermediate steps
    intermediate_steps = strip_ansi_codes(verbose_logs)
    final_answer = response.get("output", "No recommendation provided.")

    return intermediate_steps, final_answer
