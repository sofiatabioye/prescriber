import streamlit as st
from initialize_agent import initialize_agent
from guideline_recommendations import get_guideline_recommendations
from parse_and_summarize_pdf import parse_and_summarize_pdf

# Initialize the agent for recommendations
agent_executor = initialize_agent()

# App title and subheader
# Add custom CSS for the subheader
st.markdown(
    """
    <style>
    .subheader {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }
    .disabled {
        color: #999 !important;
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and styled subheader
st.title("Rapid CRC Pathway Tool")
st.markdown('<div class="subheader">Streamlined Urgent Cancer Referrals</div>', unsafe_allow_html=True)

# Dropdown selector for cancer type with disabled options
selected_cancer = st.selectbox(
    "Select Cancer Type",
    options=["Colorectal Cancer", "Bowel Cancer (Disabled)", "Other Cancer (Disabled)"],
    index=0,
    help="Currently, only Colorectal Cancer is supported."
)

# Logic to prevent interaction with disabled options
if "Disabled" in selected_cancer:
    st.warning("This option is currently not supported. Please select 'Colorectal Cancer (Enabled)'.")
    st.stop()

# Sidebar for file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a patient's 2WW referral form (PDF)", type="pdf")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat messages

for chat in st.session_state["chat_history"]:
    st.write(chat)

# File upload and analysis
if uploaded_file:
    st.sidebar.success("File uploaded successfully!")
    with st.spinner("Processing the uploaded PDF..."):
        try:
            # Parse and summarize the PDF
             # Parse and summarize the PDF
            summary = parse_and_summarize_pdf(uploaded_file)
            st.subheader("Extracted Patient Data")
            st.text_area("Summary of Patient 2WW Form", summary, height=300)

            # Get recommendations and intermediate steps
            intermediate_steps, final_answer = get_guideline_recommendations(summary, agent_executor)

            # Display final recommendation
            st.subheader("Recommendations")
            st.text_area("Detailed Recommendation", final_answer, height=150)

            # Display intermediate steps
            st.subheader("Intermediate Steps")
            st.text_area("Agent's Thought Process", intermediate_steps, height=300)


        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

st.sidebar.info("Remember to clear your chat history periodically for privacy.")
