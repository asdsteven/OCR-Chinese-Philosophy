from datetime import datetime
import sys
import os

import cv2
import numpy as np
import density


preemble = r"""\documentclass[12pt]{ctexbook}
\usepackage{enumitem}
\usepackage{unicode-math}
\usepackage[top=10mm,left=2mm,bottom=2mm,right=2mm,headsep=2mm,paperwidth=128mm,paperheight=170mm]{geometry}
\setlength{\headheight}{14pt}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\ctexset{
  contentsname = 目次,
  appendixname = 附錄,
  appendix/number = {},
  part = {
    name = {第,部},
    number = \chinese{part},
    aftername = {\quad},
    format = {\Huge\kaishu\centering},
    pagestyle = plain
  },
  chapter = {
    name = {第,章},
    number = \chinese{chapter},
    aftername = {\quad},
    format = {\LARGE\kaishu},
    pagestyle = fancy,
    beforeskip = 0pt,
    afterskip = 20pt
  },
  section = {
    name = {第,節},
    number = \chinese{section},
    aftername = {\quad},
    format = {\large\kaishu}
  },
  subsection = {
    name = {第,段},
    number = \chinese{subsection},
    aftername = {\quad},
    format = {\kaishu},
    beforeskip = 2ex plus 1ex minus .2ex
  }
}
\renewenvironment{quotation} {\list{}{
    \listparindent 0mm
    \itemindent \listparindent
    \leftmargin 12mm
    \rightmargin 0mm
    \parsep 1mm
  } \item\relax} {\endlist}
\usepackage{graphicx}

"""


def write_header(tex, matter, page_number, media_row, text_width):
    if matter == "frontmatters":
        s_page_number = f"({page_number})"
    else:
        s_page_number = f"{page_number}"
    _, header_bound, header_box_texts = media_row
    if page_number % 2 == 0:
        left_margin = header_bound[2]
        if not header_box_texts[0][1].startswith(s_page_number):
            print(f"+ page {page_number} header {" ".join(t for _, t in header_box_texts)}")
        print("".join(t for _, t in header_box_texts).removesuffix(s_page_number).strip())
        tex.writeln(r"\newpage" + f" % {page_number}")
        tex.writeln()
    else:
        for box, text in header_box_texts:
            if box[3] < 1550:
                left_margin = box[3] - text_width
        if not header_box_texts[-1][1].endswith(s_page_number):
            print(f"+ page {page_number} header {" ".join(t for _, t in header_box_texts)}")
        mark = "".join(t for _, t in header_box_texts).removesuffix(s_page_number.strip())
        tex.writeln(r"\newpage\markright{" + density.normalize_header_mark(mark) + "}" + f" % {page_number}")
        tex.writeln()
    return left_margin


class State:
    def __init__(self, prev_page_state, plugins):
        self.prev_page_state = prev_page_state
        self.plugins = plugins
        self.state = None
        self.data = []

    def __str__(self):
        return str(self.state)

    def pop_all(self, tex):
        while self.state:
            self.pop(tex)

    def pop(self, tex):
        state = self.state
        self.state = ""
        if state == "paragraph":
            tex.writeln()
        elif state == "quotation":
            tex.writeln(r"\end{quotation}")
            tex.writeln()
        elif state == "chapter":
            tex.writeln("}")
            tex.writeln()
        elif state == "section*":
            tex.writeln(r"\section*{" + "".join(self.data) + "}")
            tex.writeln(r"\addcontentsline{toc}{section}{" + "".join(self.data) + "}")
            tex.writeln()
            self.data = []
        elif state == "maybe section*":
            self.paragraph(tex, "".join(self.data), indent=False)
            self.data = []
        elif state == "itemize":
            tex.writeln(r"\end{itemize}")
            tex.writeln()
        elif state == "maybe itemize":
            for n, c in self.data[:-1]:
                tex.writeln(n + c)
                tex.writeln()
            self.paragraph(tex, "".join(self.data[-1]))
            self.data = []
        elif state == "quotation itemize":
            tex.writeln(r"\end{itemize}")
            tex.writeln()
            self.state = "quotation"
        elif state == "quotation maybe itemize":
            for n, c in self.data[:-1]:
                tex.writeln(n + c)
                tex.writeln()
            tex.writeln("".join(self.data[-1]))
            self.state = "quotation"
            self.data = []

    def paragraph(self, tex, sentence, indent=True):
        self.state = "paragraph"
        if not indent:
            tex.write(r"\noindent ")
        tex.writeln(sentence)

    def quotation(self, tex):
        tex.writeln(r"\begin{quotation}\kaishu\setlength{\parskip}{0pt}")
        self.state = "quotation"

    def maybe_itemize(self, number, content):
        self.state = "maybe itemize"
        self.data = [(number, content)]

    def quotation_maybe_itemize(self, number, content):
        self.state = "quotation maybe itemize"
        self.data = [(number, content)]

    def itemize(self, tex):
        tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=5.5em]")
        for n, c in self.data:
            tex.writeln(r"\item[" + n + "] " + c)
        self.state = "itemize"

    def quotation_itemize(self, tex):
        tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=1.5em]")
        for n, c in self.data:
            tex.writeln(r"\item[" + n + "] " + c)
        self.state = "quotation itemize"

    def run(self, tex, matter, indent_diff, tab, right_indent, prev_right_indent, texts):
        state = self.state
        if tab == 1:
            sentence = r" \quad ".join(texts)
            if "close quotation" in self.plugins and state == "quotation":
                self.pop(tex)
                self.paragraph(tex, sentence, False)
            elif state == "paragraph":
                tex.writeln(sentence)
            elif state == "maybe itemize":
                self.pop(tex)
                tex.writeln(sentence)
            elif state == "itemize":
                self.pop(tex)
                self.paragraph(tex, sentence, False)
            elif state == "maybe section*":
                self.pop(tex)
                tex.writeln(sentence)
            elif state:
                raise Exception(f"tab 1 paragraph follow {state} closely {texts}")
            elif state is None and texts[0] == self.plugins.get("chapter", None):
                tex.writeln(r"\chapter{" + sentence + "}")
                tex.writeln()
            elif state is None and density.match_chapter(texts[0], "第", "章"):
                self.state = "chapter"
                tex.write(r"\chapter{" + r" \quad ".join(texts[1:]))
            elif state is None and texts[0] == "附錄":
                self.state = "chapter"
                tex.writeln(r"\appendix")
                tex.writeln()
                tex.write(r"\chapter{" + r" \quad ".join(texts[1:]))
            elif texts == ["引", "言"]:
                tex.writeln(r"\section*{引 \quad 言}")
                tex.writeln(r"\addcontentsline{toc}{section}{引 \quad 言}")
                tex.writeln()
            elif density.match_chapter(texts[0], "第", "節"):
                self.state = "chapter"
                tex.write(r"\section{" + r" \quad ".join(texts[1:]))
            elif density.split_chapter(texts[0])[0].endswith("段"):
                self.state = "chapter"
                tex.write(r"\subsection{" + density.split_chapter(texts[0])[1] + r" \quad ".join(texts[1:]))
            elif matter == "appendix" and density.split_number(texts[0])[0]:
                if right_indent > 20:
                    self.state = "section*"
                    self.data = [sentence]
                else:
                    self.state = "maybe section*"
                    self.data = [sentence]
            else:
                self.paragraph(tex, sentence, False)
        elif tab == 2:
            sentence = r" \quad ".join(texts)
            number, content = density.split_number(sentence)
            if not number:
                number, content = density.split_chapter(sentence)
            if not number and sentence.startswith("附識"):
                number, content = density.split_number(sentence[2:])
                number = sentence[:2] + number
            arabic_number, arabic_content = density.split_number(sentence, patterns=["0123456789"])
            if "close quotation" in self.plugins and state == "quotation":
                self.pop(tex)
                self.paragraph(tex, sentence)
            elif "close chapter" in self.plugins and state == "chapter":
                self.pop(tex)
                self.paragraph(tex, sentence)
            elif "tab 2 quotation itemize" in self.plugins and not state and arabic_number and arabic_content:
                self.quotation(tex)
                tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=1.2em]")
                tex.writeln(r"\item[" + arabic_number + "] " + arabic_content + " % plugin tab 2 quotation itemize")
                self.state = "quotation itemize"
            elif "tab 2 quotation itemize" in self.plugins and state == "quotation itemize" and arabic_number and arabic_content:
                tex.writeln(r"\item[" + arabic_number + "] " + arabic_content + " % plugin tab 2 quotation itemize")
            elif "tab 2 quotation itemize" in self.plugins and state == "quotation" and arabic_number and arabic_content:
                tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=1.2em]")
                tex.writeln(r"\item[" + arabic_number + "] " + arabic_content + " % plugin tab 2 quotation itemize")
                self.state = "quotation itemize"
            elif not state:
                if number and content:
                    self.maybe_itemize(number, content)
                else:
                    self.paragraph(tex, sentence)
            elif state == "paragraph":
                if number and content:
                    self.pop(tex)
                    self.maybe_itemize(number, content)
                else:
                    self.pop(tex)
                    self.paragraph(tex, sentence)
            elif state == "maybe itemize":
                if number and content:
                    self.data.append((number, content))
                else:
                    self.pop_all(tex)
                    self.paragraph(tex, sentence)
            elif state == "itemize":
                if number and content:
                    tex.writeln(r"\item[" + number + "] " + content)
                else:
                    self.pop(tex)
                    self.paragraph(tex, sentence)
            else:
                raise Exception(f"tab 2 paragraph follow {state} closely {texts}")
        elif tab == 3:
            sentence = r" \quad ".join(texts)
            number, content = density.split_number(sentence)
            arabic_number, arabic_content = density.split_number(sentence, patterns=["0123456789"])
            if "close quotation" in self.plugins and state == "paragraph":
                self.pop(tex)
                self.quotation(tex)
                tex.writeln(sentence)
            elif "tab 2 quotation itemize" in self.plugins and state == "quotation itemize":
                if arabic_number and arabic_content:
                    tex.writeln(r"\item[" + arabic_number + "] " + arabic_content + " % plugin tab 2 quotation itemize")
                else:
                    tex.writeln(sentence)
            elif "tab 2 quotation itemize" in self.plugins and state is None and self.prev_page_state == "quotation itemize":
                self.quotation(tex)
                tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=1.2em]")
                if arabic_number and arabic_content:
                    tex.writeln(r"\item[" + arabic_number + "] " + arabic_content + " % plugin tab 2 quotation itemize")
                else:
                    tex.writeln(r"\item[] " + sentence + " % plugin tab 2 quotation itemize")
                self.state = "quotation itemize"
            elif "tab 2 quotation itemize" in self.plugins and not state and arabic_number and arabic_content:
                self.quotation(tex)
                tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=1.2em]")
                tex.writeln(r"\item[" + arabic_number + "] " + arabic_content + " % plugin tab 2 quotation itemize")
                self.state = "quotation itemize"
            elif indent_diff < -12 and state in [None, "paragraph", "itemize"] and number and content:
                # See 心體與性體第一冊 page 89, 636
                print(f"tab 3 follow {state} closely; treat as itemize.")
                if state != "itemize":
                    self.pop(tex)
                    self.itemize(tex)
                tex.writeln(r"\item[" + number + "] " + content)
            elif state == "paragraph" and prev_right_indent < 20:
                print(f"tab 3 follow paragraph closely; treat as continuation.")
                tex.writeln(sentence)
            elif not state:
                if number and content:
                    self.quotation(tex)
                    self.quotation_maybe_itemize(number, content)
                else:
                    self.quotation(tex)
                    tex.writeln(sentence)
            elif state == "maybe itemize":
                self.itemize(tex)
                tex.writeln(sentence)
            elif state == "itemize":
                tex.writeln(sentence)
            elif state == "quotation":
                if prev_right_indent > 20:
                    tex.writeln()
                if number and content:
                    self.quotation_maybe_itemize(number, content)
                else:
                    tex.writeln(sentence)
            elif state == "quotation maybe itemize":
                if number and content:
                    self.data.append((number, content))
                else:
                    self.pop(tex)
                    tex.writeln(sentence)
            elif state == "quotation itemize":
                if number and content:
                    tex.writeln(r"\item[" + number + "] " + content)
                else:
                    self.pop(tex)
                    tex.writeln(sentence)
            elif state == "maybe section*":
                # special case on page 678
                self.state = "section*"
                self.data.append(sentence)
            else:
                raise Exception(f"tab 3 quotation follow {state} closely {texts}")
        elif tab == 5 and len(texts) == 3 and all(len(x) == 1 for x in texts) and '*' in texts:
            self.pop_all(tex)
            tex.writeln(r"\hfill*\hfill*\hfill*\hfill\mbox{}")
            tex.writeln()
        elif tab == 4 or tab == 5:
            sentence = r" \quad ".join(texts)
            if "tab 2 quotation itemize" in self.plugins and state is None and self.prev_page_state in ["quotation maybe itemize", "quotation itemize"]:
                self.quotation(tex)
                tex.writeln(r"\begin{itemize}[nosep, topsep=0pt, partopsep=0pt, leftmargin=1.2em]")
                tex.writeln(r"\item[] " + sentence + " % plugin tab 2 quotation itemize continued")
                self.state = "quotation itemize"
            elif not state:
                if state is None and self.prev_page_state in ["maybe itemize", "itemize"]:
                    self.itemize(tex)
                    tex.writeln(r"\item[] " + sentence + " % continued maybe itemize")
                elif state is None and self.prev_page_state in ["quotation maybe itemize", "quotation itemize"]:
                    self.quotation(tex)
                    self.quotation_itemize(tex)
                    tex.writeln(r"\item[] " + sentence + " % continued quotation maybe itemize")
                else:
                    self.quotation(tex)
                    tex.writeln(r"\quad " + sentence + " % indented quotation heading")
            elif state == "maybe itemize":
                self.itemize(tex)
                tex.writeln(sentence)
            elif state == "itemize":
                tex.writeln(sentence)
            elif state == "quotation":
                if prev_right_indent > 20:
                    tex.writeln()
                tex.writeln(sentence)
            elif state == "quotation maybe itemize":
                self.quotation_itemize(tex)
                tex.writeln(sentence)
            elif state == "quotation itemize":
                tex.writeln(sentence)
            elif state == "chapter":
                tex.write(sentence)
            elif state == "section*":
                self.data.append(sentence)
            else:
                raise Exception(f"tab {tab} follow {state} closely {texts}")
        else:
            self.pop_all(tex)
            tex.write(r"\begin{flushright}")
            tex.write(r" \quad ".join(texts))
            tex.writeln(r"\end{flushright}")
            tex.writeln()


def write_page(tex, matter, page_number, media_rows, left_margin, tabstopper, image_black, prev_page_state, plugins={}):
    text_rows = [box_texts for media, _, box_texts in media_rows if media == "text"]
    left_margin = tabstopper.normalize_left_margin(left_margin, text_rows)
    for d, text in tabstopper.tab_outliers(left_margin, text_rows):
        if d > 5:
            print(f"tab outlier: {d} {text}")

    prev_bound = (0, 0, 0, 0)
    state = State(prev_page_state, plugins)
    prev_right_indent = None
    for media, bound, box_texts in media_rows:
        if media == "figure":
            state.pop_all(tex)
            trim, size, filename = box_texts
            tex.write(r"""\begin{center}
\noindent\includegraphics[clip, trim=""" + trim + ", " + size + "]{" + filename + r"""}
\end{center}

""")
            continue

        texts = [t for _, t in box_texts if t]

        indent, tab, right_indent, _ = tabstopper.indent_tab(left_margin, box_texts)
        if indent is None or tab is None:
            print(f"<empty row {bound}>")
            continue
        tight_bound = density.tight_bound(image_black, box_texts, 0.01)
        if prev_bound is not None and tight_bound[0] - prev_bound[1] > 50:
            state.pop_all(tex)
        print(f"    ↕{tight_bound[0] - prev_bound[1]:3d} {indent: 3d}:{tab} {right_indent:4d} {"  " * round(indent / 30)} {state} {texts}")
        prev_bound = tight_bound

        state.run(tex, matter, indent - tabstopper.tabstops[tab], tab, right_indent, prev_right_indent, texts)
        prev_right_indent = right_indent
    prev_page_state = state.state
    state.pop_all(tex)
    return prev_page_state


