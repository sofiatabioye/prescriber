# load_queries.py

def load_queries(file_path):
    """
    Load queries from a text file and return them as a list of strings.
    Each query is stripped of trailing newlines or whitespace.
    """
    try:
        with open(file_path, 'r') as file:
            queries = file.readlines()
        # Remove any trailing newline characters
        return [query.strip() for query in queries if query.strip()]
    except FileNotFoundError:
        return [f"Error: {file_path} not found. Please make sure the file is available."]

# Example usage if this file is imported:
# lynch_queries = load_queries("lynch.txt")
# general_queries = load_queries("queries.txt")

# You can use these returned lists in your Streamlit app to display in the sidebar or elsewhere.
