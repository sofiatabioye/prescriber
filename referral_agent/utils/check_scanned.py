import fitz  # PyMuPDF

def check_pdf_for_scanned_images(pdf_path):
    with fitz.open(pdf_path) as pdf:
        has_images = False
        has_text = False

        for page in pdf:
            # Check if the page has selectable text
            if page.get_text():
                has_text = True
            
            # Check if the page contains images
            images = page.get_images(full=True)
            if images:
                has_images = True

        return has_text, has_images

# Example usage
file_path = "form1.pdf"  # Replace with your file path
text_present, images_present = check_pdf_for_scanned_images(file_path)

if images_present and not text_present:
    print("The PDF contains scanned images but no selectable text.")
elif text_present and not images_present:
    print("The PDF contains selectable text and no scanned images.")
elif images_present and text_present:
    print("The PDF contains both scanned images and selectable text.")
else:
    print("The PDF is empty or contains unsupported content.")
