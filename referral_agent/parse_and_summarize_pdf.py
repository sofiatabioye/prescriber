### Module 2: PDF Processing and Summarization ###
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from ollama_llama import OllamaLlama


def extract_text_with_ocr(pdf_path):
    """Extracts text from a PDF using OCR for scanned pages."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            page_text = page.get_text()
            
            if not page_text.strip():  # If no selectable text, apply OCR
                images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
                for image in images:
                    page_text += pytesseract.image_to_string(image)
            
            text += page_text
    return text

def parse_and_summarize_pdf(file_path):
    """Parses and summarizes a PDF file with mixed content."""
    # Step 1: Extract text (handles both selectable text and scanned images)
    raw_text = extract_text_with_ocr(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    split_docs = text_splitter.split_text(raw_text)
    
    # llm = OllamaLlama(model="llama-2-7b-chat");
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    summaries = []
    for chunk in split_docs:
        prompt = f"""
            This text is from a 2WW referral form for colorectal cancer. Extract the following details:
            1. Patient details (e.g., name, age, gender, address, hospital number).
            2. GP declaration (if present)
            3. GP/Doctor and referral date details
            4. Symptoms (e.g., abdominal mass, rectal bleeding).
            5. FIT result (positive or negative, and value if available).
            6. FIT positive pathway results (tick-marked if present and explain the results recorded).
            7. FIT negative patients with Iron Deficiency Anaemia results (tick-marked if present).
            8. Additional history (if mentioned).
            9. WHO performance status (on the WHO scale, tick-marked value).

            Text:
            {chunk}
            """
        response = llm.invoke(prompt)
        print(response)
        summaries.append(response.content if hasattr(response, 'content') else str(response))

    return " ".join(summaries)

# Example Usage
if __name__ == "__main__":
    file_path = "form2.pdf"  # Replace with the path to your PDF file
    summarized_text = parse_and_summarize_pdf(file_path)
    print("Summarized Content:")
    print(summarized_text)
