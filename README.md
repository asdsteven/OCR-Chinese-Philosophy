# OCR Chinese Philosophy

This project aims to perform OCR on 牟宗三's work to produce Pandoc Markdown.  Only the Markdown files are provided, but not the scanned copy of the work.

In general, each PDF scanned copy is first passed to `mutool` to extract each page as images.  Next, we run the main ocr program.  The program would first compute text boxes location, which includes reducing noise and is highly specific to each book, so a separate OCR program will be made for each book.  Then, the program passes the text boxes to PaddleOCR (PP‑OCRv5_server_rec) to get the text.  Finally, the text is organized as Pandoc Markdown on LaTeX pdfengine.

Currently, two fonts are being used to produce PDF from Pandoc Markdown: "Chiron Sung HK" and "DFHKStdKai-B5".

I initiated this project when I was about to read 心體與性體 deeply.  OCR becomes a handy byproduct.

---

## Usage

Run OCR:
```bash
python ocr.py
```

It produces `ocr.md` which contains the main result, and `crosscheck-ocr.md` which additionally interleaves original page for manual crosschecking.  It takes ~1 hour to process all 700 pages.

Convert to PDF:
```bash
pandoc ocr.md -o ocr.pdf --pdf-engine=xelatex
```

