from parse_and_summarize_pdf import parse_and_summarize_pdf

### Module 3: Recommendation Generation ###

def get_guideline_recommendations(file_path, agent_executor):
    """Generates recommendations based on parsed PDF content."""
    summary = parse_and_summarize_pdf(file_path)
    prompt = f"""
    Based on the following patient data extracted from a medical document.
    {summary}

    Please provide guideline-based recommendations for the patient's 2WW referral form. Use the medical guidelines stored in the system to inform your response.
    """

    response = agent_executor.invoke({"input": prompt, "chat_history": ""})
    return response