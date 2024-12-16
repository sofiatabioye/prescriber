from initialize_agent import initialize_agent
from guideline_recommendations import get_guideline_recommendations

### Example Usage ###
if __name__ == "__main__":
    file_path = "form5.pdf"
    agent_executor = initialize_agent()
    recommendations = get_guideline_recommendations(file_path, agent_executor)

    print("Recommendations:")
    print(recommendations)
