from initialize_agent import initialize_agent
from guideline_recommendations import get_guideline_recommendations
from parse_and_summarize_pdf import parse_and_summarize_pdf

### Example Usage ###
if __name__ == "__main__":
    # Open the PDF file in binary mode
    file_path = "sample_forms/form3.pdf"
    with open(file_path, "rb") as pdf_file:
        # Initialize the agent
        agent_executor = initialize_agent()

        # Parse and summarize the PDF
        summary = parse_and_summarize_pdf(pdf_file)

        # Generate recommendations
        recommendations = get_guideline_recommendations(summary, agent_executor)

        # Print results
        print("Extracted Summary:")
        print(summary)
        print("\nRecommendations:")
        print(recommendations)
