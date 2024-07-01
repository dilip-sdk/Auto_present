import os
import time

import pptx
from PIL import Image
import fitz  # PyMuPDF for PDFs
# import python_pptx
# import pptx2pdf  # for converting ppt/pptx to PDF (requires pptx2pdf library)


def resize_if_needed(image, max_width=1920, max_height=1080):
    width, height = image.size
    print(height)
    if width <= max_width and height <= max_height:
        image = image.resize((max_width, max_height))
    return image


def convert_ppt_to_pdf(ppt_path, pdf_path):
    pptx2pdf.convert(ppt_path, pdf_path)


def save_document_pages_as_images(doc_path, output_folder):
    start_time = time.time()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine document type and handle accordingly
    file_extension = os.path.splitext(doc_path)[1].lower()

    if file_extension == ".pdf":
        doc = fitz.open(doc_path)
        total_pages = doc.page_count

        for page_number in range(total_pages):
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = resize_if_needed(img)

            output_file_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
            img.save(output_file_path, "PNG")
            print(f"Saved PDF page {page_number + 1} as {output_file_path}")

    elif file_extension in (".ppt", ".pptx"):
        pdf_path = os.path.join(output_folder, "temp.pdf")
        convert_ppt_to_pdf(doc_path, pdf_path)
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        for page_number in range(total_pages):
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = resize_if_needed(img)

            output_file_path = os.path.join(output_folder, f"slide_{page_number + 1}.png")
            img.save(output_file_path, "PNG")
            print(f"Saved PowerPoint slide {page_number + 1} as {output_file_path}")

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"All pages/slides have been saved as images. Time taken: {elapsed_time:.2f} seconds.")


# Example usage:
doc_path = ""  # Replace with the path to your document file (PDF, PPT, etc.)
output_folder = "sample_output"  # Replace with your desired output folder
# save_document_pages_as_images(doc_path, output_folder)
print(len(doc_path))