"""
ingestion.py
============
Document ingestion module — loads text and PDF files.
"""

import os
import re


def load_text_file(filepath: str) -> str:
    """Load plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(filepath: str) -> str:
    """Load PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        return ""
    except Exception as e:
        return ""


def load_documents(file_paths: list) -> str:
    """
    Load multiple documents (txt or pdf) and combine into one text.
    """
    combined = ""
    for path in file_paths:
        if not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            combined += load_text_file(path) + "\n\n"
        elif ext == ".pdf":
            combined += load_pdf_file(path) + "\n\n"
    return combined


def extract_paragraphs(text: str, min_length: int = 80) -> list:
    """Split text into meaningful paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > min_length]
    return paragraphs
