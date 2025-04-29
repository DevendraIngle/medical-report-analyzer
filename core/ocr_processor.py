import base64
import io
import pypdfium2 as pdfium
from PIL import Image
import streamlit as st
from openai import OpenAI
#import fitz # PyMuPDF - Used for PDF page count and basic text check if needed

from utils.helpers import image_to_base64

class OCRProcessor:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API key is not provided.")
        self.client = OpenAI(api_key=openai_api_key)

    def extract_text_from_image(self, image_bytes):
        """Extracts text from image bytes using OpenAI GPT-4 Vision."""
        base64_image = image_to_base64(image_bytes)

        try:
            st.spinner("Sending image to OpenAI Vision for OCR...")
            response = self.client.chat.completions.create(
                model="gpt-4o", # Using gpt-4o
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this medical report image. Preserve formatting and newlines as much as possible."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high" # Request high detail
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
            )
            text = response.choices[0].message.content
            st.success("Image OCR complete.")
            return text if text else ""
        except Exception as e:
            st.error(f"Error during image OCR: {e}")
            return None

    def extract_text_from_pdf(self, pdf_bytes):
        """Extracts text from PDF using PyPDFium2 (text first, then Vision fallback)."""
        all_text = ""
        pages_to_process_with_vision = []
        temp_file_path = None

        try:
            # Use PyMuPDF to quickly check number of pages and initial text extraction
            # Using PyPDFium2 for robust image rendering
            pdf_document = pdfium.PdfDocument(pdf_bytes)
            num_pages = len(pdf_document)

            st.info(f"Processing PDF with {num_pages} pages...")

            for page_number in range(num_pages):
                page = pdf_document.get_page(page_number)
                try:
                    # Attempt text extraction using PyPDFium2
                    text = page.get_text()
                    # A simple heuristic: if text length is very short, assume it's image-based or complex layout
                    if len(text.strip()) < 100: # Adjusted threshold
                         st.warning(f"Page {page_number + 1} yielded limited text. Preparing for Vision OCR...")
                         # Render page as image for Vision fallback
                         # Higher DPI (e.g., 300 DPI = scale 300/72 = ~4.16) for better OCR
                         pil_image = page.render(scale=300/72).to_image()
                         img_byte_arr = io.BytesIO()
                         # Save as PNG to preserve quality for Vision
                         pil_image.save(img_byte_arr, format='PNG')
                         pages_to_process_with_vision.append((page_number + 1, img_byte_arr.getvalue()))
                         all_text += f"\n--- Page {page_number + 1} (Processing with Vision) ---\n" # Placeholder note
                    else:
                        all_text += f"\n--- Page {page_number + 1} ---\n" + text

                except Exception as page_e:
                    st.warning(f"Error processing page {page_number + 1} for text, trying Vision: {page_e}")
                    # Render page as image for Vision fallback
                    pil_image = page.render(scale=300/72).to_image()
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    pages_to_process_with_vision.append((page_number + 1, img_byte_arr.getvalue()))
                    all_text += f"\n--- Page {page_number + 1} (Processing with Vision) ---\n" # Placeholder note


            # Process pages marked for Vision OCR
            if pages_to_process_with_vision:
                 st.info(f"Performing Vision OCR on {len(pages_to_process_with_vision)} page(s)... This might take some time.")
                 for page_num, img_bytes in pages_to_process_with_vision:
                     st.write(f"Processing page {page_num} with Vision...")
                     vision_text = self.extract_text_from_image(img_bytes)
                     if vision_text:
                         # Replace the placeholder note with the actual Vision text
                         placeholder = f"\n--- Page {page_num} (Processing with Vision) ---\n"
                         all_text = all_text.replace(placeholder, f"\n--- Page {page_num} (Vision OCR) ---\n" + vision_text, 1)
                     else:
                         st.error(f"Vision OCR failed for page {page_num}.")
                         placeholder = f"\n--- Page {page_num} (Processing with Vision) ---\n"
                         all_text = all_text.replace(placeholder, f"\n--- Page {page_num} (Vision OCR Failed) ---\n", 1)


            if not all_text.strip() or all_text.strip().replace("--- Page", "").strip() == "":
                 st.warning("Could not extract significant text from any page of the PDF.")
                 return None


            st.success("PDF text extraction complete.")
            return all_text.strip()

        except Exception as e:
            st.error(f"Error during PDF processing: {e}")
            return None

    def process(self, file_bytes, file_type):
        """Processes uploaded file bytes based on type."""
        if file_type in ["image/png", "image/jpeg"]:
            st.info("Processing image file...")
            return self.extract_text_from_image(file_bytes)
        elif file_type == "application/pdf":
            st.info("Processing PDF file...")
            return self.extract_text_from_pdf(file_bytes)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None

    def get_pdf_images(self, pdf_bytes):
        """Generates images for each page of a PDF for preview."""
        images = []
        try:
            pdf_document = pdfium.PdfDocument(pdf_bytes)
            num_pages = len(pdf_document)
            for page_number in range(num_pages):
                page = pdf_document.get_page(page_number)
                # Render at a lower scale for quick preview
                pil_image = page.render(scale=1).to_image()
                images.append(pil_image)
            return images
        except Exception as e:
            st.error(f"Error rendering PDF pages for preview: {e}")
            return []