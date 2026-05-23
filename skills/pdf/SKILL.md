---
name: pdf
description: Extract and process PDF text content
tags: pdf, parsing, file
---
# PDF Processing Skill
## Extract text from PDF
Use command:
pdftotext input.pdf -

## Python library
Install:
pip install pymupdf

Simple code:
import fitz
doc = fitz.open("input.pdf")
for page in doc:
    print(page.get_text())