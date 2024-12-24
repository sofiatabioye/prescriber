### Module 2: PDF Processing and Summarization ###
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO

def extract_text_with_ocr(file):
    """
    Extracts text from a PDF using both PyMuPDF for selectable text and OCR for scanned pages.
    Accepts a file-like object (BytesIO).
    """
    text = ""
    try:
        # Open the file using PyMuPDF
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()

            if page_text.strip():
                text += page_text
            else:
                # Use OCR if no selectable text
                pdf_document.seek(0)  # Reset file pointer for pdf2image
                images = convert_from_bytes(file.read(), first_page=page_num + 1, last_page=page_num + 1)
                for image in images:
                    text += pytesseract.image_to_string(image)
    except Exception as e:
        raise ValueError(f"Error processing the PDF file: {e}")
    return text

def parse_and_summarize_pdf(file):
    """
    Parses and summarizes a PDF file with mixed content (selectable text and scanned images).
    Accepts a file-like object (BytesIO).
    """
    # Step 1: Extract text
    raw_text = extract_text_with_ocr(file)
    
    # Step 2: Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
    split_docs = text_splitter.split_text(raw_text)

    # Step 3: Use LLM for summarization
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
    )
    summaries = []

    for chunk in split_docs:
        prompt = f"""
            This text is from a 2WW referral form for colorectal cancer. Extract the following details and format the output as a structured summary. Use the rules below to ensure accuracy:
            
            0. ** FIT Result
            - If the FIT result is less than 10 ugHb/g, this will be recorded as Negative, while greater than or equal to 10 ugHb/g will be Positive.
            
            1. **FIT Positive Pathway Results Rules**:
            - For each parameter in the FIT positive pathway results (Rectal bleeding, Change in bowel habit, Weight loss, Iron Deficiency Anaemia), check if it is mentioned in the text.
            - If the parameter is present and has a FIT value (e.g., 'FIT result: ...'), mark it as 'Yes' and include the FIT value.
            - If the parameter is present without a FIT value, mark it as 'Yes' but indicate 'FIT value: Not provided'.
            - If the parameter is not mentioned in the text, mark it as 'No'.

            2. **FIT Negative Pathway Results Rules**:
            - For FIT-negative patients with Iron Deficiency Anaemia (IDA), extract the following:
                - Indicate if the patient meets all criteria for referral:
                - Aged 40 years or over: Yes/No
                - FIT Negative: Yes/No (This is a "Yes" if FIT result is less than 10), FIT result: [Value/Not provided]
                - Ferritin ≤45 µg/L: Yes/No (This is a "Yes" if Ferritin value ≤45µg/L), Ferritin: [Value/Not provided]
                - Anaemia: Yes/No (Yes if Hb value less than 130 g/L (13 g/dL) in men or 115 g/L (11.5 g/dL) in non-menstruating women). Ensure that Hb values reported in g/L are compared to thresholds directly in g/L, Hb: [Value/Not provided].
                
            3. **General Formatting Rules**:
            - Provide a structured summary with the fields below.
            - Use the following format for the output:
                - Name: [Value]
                - Age: [Value]
                - Gender: [Value]
                - Address: [Value]
                - Hospital number: [Value]
                - GP declaration: [Value]
                - GP/Doctor details and referral date: [Value]
                - Symptoms: [Value]
                - FIT result: [Value], FIT [Positive/Negative]
                - FIT positive pathway results:
                    - Rectal bleeding: [Yes/No], FIT result: [Value/Not provided]
                    - Change in bowel habit: [Yes/No], FIT result: [Value/Not provided]
                    - Weight loss: [Yes/No], FIT result: [Value/Not provided]
                    - Iron Deficiency Anaemia: [Yes/No], FIT result: [Value/Not provided]
                - FIT negative pathway results:
                    - Meets criteria for referral: [Yes/No]
                    - Aged 40 years or over: [Yes/No]
                    - FIT Negative: [Yes/No], FIT result: [Value/Not provided]
                    - Ferritin ≤45 µg/L: [Yes/No], Ferritin: [Value/Not provided]
                    - Anaemia: [Yes/No], Hb: [Value/Not provided]
                - WHO Performance status: [Value]
                - Additional History: [Value]

            Text:
            {chunk}
            """
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)
        summaries.append(result)

    # Format the final structured summary
    formatted_summary = "\n".join(summaries)
    return formatted_summary


# from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import fitz  
# from pdf2image import convert_from_path
# import pytesseract
# # from ollama_llama import OllamaLlama
# from langchain_community.llms import Ollama
# from langchain_ollama import OllamaLLM

# def extract_text_with_ocr(pdf_path):
#     """Extracts text from a PDF using OCR for scanned pages."""
#     text = ""
#     with fitz.open(pdf_path) as pdf:
#         for page_num in range(len(pdf)):
#             page = pdf[page_num]
#             page_text = page.get_text()
            
#             if not page_text.strip():  # If no selectable text, apply OCR
#                 images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
#                 for image in images:
#                     page_text += pytesseract.image_to_string(image)
            
#             text += page_text
#     return text

# def parse_and_summarize_pdf(file_path):
#     """Parses and summarizes a PDF file with mixed content."""
#     # Step 1: Extract text (handles both selectable text and scanned images)
#     raw_text = extract_text_with_ocr(file_path)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
#     split_docs = text_splitter.split_text(raw_text)
    

#     llm = ChatOpenAI(
#         model="gpt-4o",
#         temperature=0.7,
#     )
#     summaries = []

#     for chunk in split_docs:
#         prompt = f"""
#             This text is from a 2WW referral form for colorectal cancer. Extract the following details and format the output as a structured summary. Use the rules below to ensure accuracy:
#             0. ** FIT Result
#             - If the FIT result is less than 10 ugHb/g, this will be recorded as Negative, while greater than or equal to 10 ugHb/g will be Positive
#             1. **FIT Positive Pathway Results Rules**:
#             - For each parameter in the FIT positive pathway results (Rectal bleeding, Change in bowel habit, Weight loss, Iron Deficiency Anaemia), check if it is mentioned in the text.
#             - If the parameter is present and has a FIT value (e.g., 'FIT result: ...'), mark it as 'Yes' and include the FIT value.
#             - If the parameter is present without a FIT value, mark it as 'Yes' but indicate 'FIT value: Not provided'.
#             - If the parameter is not mentioned in the text, mark it as 'No'.

#             2. **FIT Negative Pathway Results Rules**:
#             - For FIT-negative patients with Iron Deficiency Anaemia (IDA), extract the following:
#                 - Indicate if the patient meets all criteria for referral:
#                 - Aged 40 years or over: Yes/No
#                 - FIT Negative: Yes/No (This is a "Yes" if FIT result is less and 10), FIT result: [Value/Not provided]
#                 - Ferritin ≤45 µg/L: Yes/No (This is a "Yes" if Ferritin value ≤45µg/L), Ferritin: [Value/Not provided]
#                 - Anaemia: Yes/No (Yes if Hb value less than 130 g/L (13 g/dL) in men or 115 g/L (11.5 g/dL) in non-menstruating women). Ensure that Hb values reported in g/L are compared to thresholds directly in g/L, Hb: [Value/Not provided]
#                 - Additional tests:
#                 - Dipstick urine: [Positive/Negative/Not provided](If positive consider referral on urology 2WW)
#                 - Screen for Coeliac disease: [Positive/Negative/Not provided](If positive refer to gastroenterology)
#                 - Renal function: [Values for Urea, Creatinine, eGFR if provided] (MUST be within 3 months)
#                 - Iron treatment: [Date commenced/Not provided]
#                 - Other results (MCV, TTG): [Values/Not provided]
                
#             3. **General Formatting Rules**:
#             - Provide a structured summary with the fields below.
#             - Use the following format for the output:
#                 - Name: [Value]
#                 - Age: [Value]
#                 - Gender: [Value]
#                 - Address: [Value]
#                 - Hospital number: [Value]
#                 - GP declaration: [Value]
#                 - GP/Doctor details and referral date: [Value]
#                 - Symptoms: [Value]
#                 - FIT result: [Value], FIT [Positive/Negative]
#                 - FIT positive pathway results:
#                     - Rectal bleeding: [Yes/No], FIT result: [Value/Not provided]
#                     - Change in bowel habit: [Yes/No], FIT result: [Value/Not provided]
#                     - Weight loss: [Yes/No], FIT result: [Value/Not provided]
#                     - Iron Deficiency Anaemia: [Yes/No], FIT result: [Value/Not provided]
#                 - FIT negative patients with Iron Deficiency Anaemia results: [Value]
#                 - FIT negative pathway results:
#                     - Meets criteria for referral: [Yes/No]
#                     - Aged 40 years or over: [Yes/No]
#                     - FIT Negative: [Yes/No], FIT result: [Value/Not provided]
#                     - Ferritin ≤45 µg/L: [Yes/No], Ferritin: [Value/Not provided]
#                     - Anaemia: [Yes/No], Hb: [Value/Not provided]
#                     - Dipstick urine: [Positive/Negative/Not provided]
#                     - Screen for Coeliac disease: [Positive/Negative/Not provided]
#                     - Renal function: Urea: [Value/Not provided], Creatinine: [Value/Not provided], eGFR: [Value/Not provided]
#                     - Iron treatment commenced: [Date/Not provided]
#                     - MCV: [Value/Not provided], TTG: [Value/Not provided]
#                 - WHO Performance status: [Value]
#                 - Additional History: [Value]

#             Text:
#             {chunk}
#             """

#         response = llm.invoke(prompt)
#         result = response.content if hasattr(response, "content") else str(response)
#         summaries.append(result)
#         # Skip irrelevant chunks
#         if "does not contain any of this specific information" in result:
#             continue
#     # Format the final structured summary
#     formatted_summary = "\n".join(summaries)
#     return formatted_summary

