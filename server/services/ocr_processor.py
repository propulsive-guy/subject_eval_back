"""
This Module performs the OCR processing
"""

"""
Inbuilt Modules 
"""
import os
import io
import json
from typing import Dict

from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_bytes


class OcrProcessor:
    """
    A class to process PDFs, perform OCR using Google's Gemini model,
    and structure the extracted text.
    """


    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the processor by configuring the API and loading the model.
        """
        self._configure_api()
        self.model = genai.GenerativeModel(model_name)
        self.prompt = self._get_ocr_prompt()



    def _configure_api(self):
        """Loads environment variables and configures the Google AI API."""
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the .env file")
        genai.configure(api_key=google_api_key)



    def _get_ocr_prompt(self) -> str:
        """Returns the standardized prompt for the OCR task."""
        return """
        You are a highly accurate OCR + exam answer extraction agent.

### GOAL
Extract student answers from provided answer sheet images and convert them into a **clean structured JSON** mapping.

### What To Do
- Read all pages carefully (scanned, tilted, handwritten allowed)
- Identify **question numbers and sub-parts**:
    Formats may appear like:
    - Q1 / Q.1 / Q-1 / Question 1
    - Q1a, Q1(A), Q1.1, Part A, (a), 1(a)
- If question numbers are missing or unclear:
    → **Infer the correct question number by context & continuity**

- **Combine multi-page answers**
    If content continues without new question heading, append it to the same question.

- **Preserve original meaning**, lists, math, bullet points, paragraph structure, and any key diagrams (describe briefly if text present near diagram)

- **Ignore noise** like:
    - margins, scribbles, cross marks, page numbers, ticks, stamps, symbols
    - handwritten shapes (unless labelled)
    - watermarks

- **VERY IMPORTANT**
    - No hallucination. Only extract what student actually wrote.
    - If confused, mark `"uncertain"` and still extract text.

### OUTPUT FORMAT (STRICT JSON)
Return only JSON like:

{
  "Q1": "full answer",
  "Q1A": "full answer",
  "Q2": "full answer",
  "Q3B": "full answer"
}

Rules:
- Keys must be normalized:
    Q1, Q1A, Q1B, Q2, Q2A, ...
- No markdown, no code block, no backticks
- No explanation text from you
- Do not invent missing answers

### Edge Handling
- If partial question labeled like "(a)" → infer previous main question: Q1A
- If you detect paragraphs without Q label but clearly continuation → append
- If student skipped question → simply omit it

Now process the answer sheet images and return structured answers only.
"""
       


    async def process_pdf(self, pdf_bytes: bytes) -> Dict[str, str]:
        """
        Converts a PDF in bytes to images and uses the Gemini model to perform OCR.

        Args:
            pdf_bytes (bytes): The content of the PDF file.

        Returns:
            A dictionary of question numbers and their corresponding text.
        """
        try:
            images = convert_from_bytes(pdf_bytes)
            if not images:
                return {}

            image_parts = []
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                image_parts.append({"mime_type": "image/png", "data": buffered.getvalue()})

            response = await self.model.generate_content_async([self.prompt] + image_parts)
            
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)

        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return {}
