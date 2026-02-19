# OCR Chinese Philosophy

This project aims to perform OCR on 牟宗三's work to produce CTeX.  Only the output files are provided, but not the scanned copy of the work.

In general, each PDF scanned copy is first passed to `mutool` to extract each page as images.  Next, we run the main ocr program.  The program would first compute text boxes location, which includes noise reduction / heading parsing and is highly specific to each book content structure, so a standalone OCR program will be made for each book.  Then, the program passes the text boxes to PaddleOCR (PP‑OCRv5_server_rec) to get the text.  Finally, the text is organized as CTeX / XeLaTeX.

Header, Part, Chapter, Section, Subsection, Appendix are parsed and produced as LaTeX native constructs like `fancyhdr`, \part`, `\chapter` etc.  Table of Content is thus generated entirely by LaTeX, and is not OCRed.

I initiated this project when I was about to read 心體與性體 deeply.  As I deep read the OCR output, I am also crosschecking it.  I find that the output is mostly correct and is never affects reading, so I don't usually bother manual correction.

---

## Usage

Run OCR:
```bash
python ocr.py
```

It produces `ocr.tex` which contains the main result, and `crosscheck-ocr.tex` which additionally interleaves original book pages for manual crosschecking.  It takes ~1 hour to generate all 700 pages.

Convert to PDF (run two times so table of contents work properly):
```bash
xelatex ocr.tex
xelatex ocr.tex
```

