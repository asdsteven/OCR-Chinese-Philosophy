from datetime import datetime
import os
import cv2
import numpy as np
from paddleocr import TextRecognition


input_dir = "ocr-input/"


buf = []
def write_file(s, flush=False, crosscheck_only=False, mode="a"):
    global buf
    if not flush:
        buf.append(s)
        return
    s = "".join(buf) + s
    buf = []
    if not crosscheck_only:
        with open("ocr.tex", mode) as file:
            file.write(s)
    with open("crosscheck-ocr.tex", mode) as file:
        file.write(s)


write_file(r"""
\documentclass[12pt]{ctexbook}
\usepackage{unicode-math}
\usepackage[top=10mm,left=2mm,bottom=0mm,right=2mm,headsep=2mm,paperwidth=128mm,paperheight=170mm]{geometry}
\setlength{\headheight}{14pt}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\chaptermark}[1]{\markboth{心體與性體 \quad 第一冊}{}}
\renewcommand{\sectionmark}[1]{\markboth{心體與性體 \quad 第一冊}{}}
\renewcommand{\subsectionmark}[1]{\markboth{心體與性體 \quad 第一冊}{}}
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
    beforeskip = 40pt,
    afterskip = 30pt
  },
  section = {
    name = {第,節},
    number = \chinese{section},
    aftername = {\quad},
    format = {\large\kaishu},
  },
  subsection = {
    name = {第,段},
    number = \chinese{subsection},
    aftername = {\quad},
    format = {\kaishu},
  }
}
\renewenvironment{quotation} {\list{}{
    \listparindent 0mm
    \itemindent \listparindent
    \rightmargin 0mm
    \parsep 1mm
  } \item\relax} {\endlist}
\usepackage{graphicx}

\begin{document}

\frontmatter

\begin{titlepage}
  \begin{flushright}
    \vspace*{\fill}
    {\heiti 牟宗三先生全集\textcircled{5}}

    \medskip

    {\Huge\songti 心體與性體}

    \smallskip

    {\kaishu （第一冊）}

    \smallskip

    {\songti 牟宗三 \quad 著}
    \vspace*{\fill}
  \end{flushright}
\end{titlepage}

\pagenumbering{arabic}
\fancyhead[LE]{\small (\thepage )\ $\odot$\ \kaishu\leftmark}
\fancyhead[RO]{\small \kaishu\rightmark\ $\odot$\ (\thepage )}
""", mode="w", flush=True)


def black_white_chunks(black_whites, link=None):
    # black_whites is array of 0 (black) and 1 (white)
    boundaries = np.where(np.diff(black_whites) != 0)[0]
    if boundaries.size == 0:
        return [(0, len(black_whites))]
    chunks = [(black_whites[0], 0, boundaries[0] + 1)]
    for i, x in enumerate(boundaries[:-1]):
        chunks.append((black_whites[x + 1], x + 1, boundaries[i + 1] + 1))
    chunks.append((black_whites[-1], boundaries[-1] + 1, len(black_whites)))
    if link is not None:
        i = 1
        while i < len(chunks) - 1:
            if chunks[i][0] == 1 and chunks[i][2] - chunks[i][1] <= link:
                # link small whiite chunks
                chunks[i - 1] = (chunks[i - 1][0], chunks[i - 1][1], chunks[i + 1][2])
                del chunks[i + 1]
                del chunks[i]
            else:
                i += 1
    return [(l, h) for (c, l, h) in chunks if c == 0]



def remove_left_right_glitches(image, new_image, manual_crop):
    for (left, right) in black_white_chunks(np.where(np.all(image == 255, axis=(0,2)), 1, 0)):
        if right - left < 5:
            print(f"    left right glitch {left}+{right - left}")
            new_image[:, left:right] = image[:, left:right]
            image[:, left:right] = (255, 255, 255)
    if manual_crop is None:
        return
    (left, right) = manual_crop
    print(f"    manual crop {left},{right}")
    new_image[:, :left] = image[:, :left]
    new_image[:, right:] = image[:, right:]
    image[:, :left] = (255, 255, 255)
    image[:, right:] = (255, 255, 255)


model = TextRecognition()
manual_crop = {
    261: (0, -50),
    263: (100, -1),
    621: (50, -1)
}
indents = {}
appendix = False


def ocr(image, page_number, str_page_number):
    global indents, appendix

    new_image = image.copy()
    new_image[:, :] = (255, 100, 0)

    remove_left_right_glitches(image, new_image, manual_crop.get(page_number))

    trims = [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]

    rows = black_white_chunks(np.where(np.all(image == 255, axis=(1,2)), 1, 0))

    # page header
    while rows:
        (top, bottom) = rows[0]
        def predict(lr):
            [left, right] = lr
            if right - left < 10:
                return None
            trims[1] = min(trims[1], image.shape[0] - bottom)
            trims[3] = min(trims[3], top)
            trims[0] = min(trims[0], left)
            trims[2] = min(trims[2], image.shape[1] - right)
            s = model.predict(input=image[top-5:bottom+10, left-5:right+10])[0]["rec_text"]
            if s.find("<") != -1 or s.find(">") != -1:
                t = s.replace("<", "〈").replace(">", "〉")
                print(f"    +replace {s} -> {t}")
                return t
            return s
        headers = black_white_chunks(np.where(np.all(image[top:bottom] == 255, axis=(0,2)), 1, 0), link=11)

        for i, (left, right) in enumerate(headers):
            new_image[top:bottom, left:right] = image[top:bottom, left:right]
            # if i > 0:
            #     print(f"    header gap {left - headers[i - 1][1]}")
            # print(f"    header chunk {right - left}")

        if bottom - top > 10:
            if page_number % 2 == 0 and len(headers) >= 4 and predict(headers[0]) == str_page_number:
                indent0 = int(headers[0][0])
                # odot = predict(1)
                book = predict(headers[2]) if len(headers) >= 3 else "-"
                volume = predict([headers[3][0], headers[-1][1]]) if len(headers) >= 4 else "-"
                if book != "心體與性體" or (volume != "第一册" and volume != "第一冊"):
                    print(f"    special even header: {book} {volume}")
                write_file("\n\n\\newpage")
                rows = rows[1:]
                break
            elif page_number % 2 == 1 and predict(headers[-1]) == str_page_number:
                trims[1] = min(trims[1], image.shape[0] - bottom)
                trims[3] = min(trims[3], top)
                trims[0] = min(trims[0], headers[0][0])
                trims[2] = min(trims[2], image.shape[1] - headers[-1][1])
                indent0 = int(headers[-1][1]) - 1268
                s = []
                for i, lr in enumerate(headers[:-2]):
                    t = predict(lr)
                    if t is None:
                        continue
                    if s and lr[0] - headers[i - 1][1] < 20 and (s[-1].endswith("〈") or s[-1].endswith("〉") or t.startswith("〈") or t.startswith("〉")):
                        s[-1] += t
                    else:
                        s.append(t)
                    if s[:] == ["第二部","分論一","第二章"]:
                        s.append("張橫渠對於「天道性命相貫通」之展示")
                        break
                write_file("\n\n\\newpage\\markright{" + " \\quad ".join(s) + "}")
                rows = rows[1:]
                break
        print(f"    noisy header {right - left}x{bottom - top}+{left}+{top}")
        rows = rows[1:]

    if not rows:
        print(f"    blank page")
        write_file("\n\n\\newpage\\thispagestyle{empty}")
        return new_image

    # page content
    prev_paragraph = None
    prev_right = None
    prev_bottom = None
    for line, (top, bottom) in enumerate(rows):
        chunks = black_white_chunks(np.where(np.all(image[top:bottom] == 255, axis=(0,2)), 1, 0), link=50)

        # print(f"    {bottom - top} height") # debug
        for i, (left, right) in enumerate(chunks):
            if i > 0:
                print(f"    word gap {left - chunks[i - 1][1]}")

        cur_paragraph = None
        cur_right = None
        cur_bottom = None
        indent_level = None
        for (left, right) in chunks:
            new_image[top:bottom, left:right] = image[top:bottom, left:right]
            if bottom - top < 10 or right - left < 10:
                print(f"            noise {right - left}x{bottom - top}+{left}+{top}")
                continue

            cur_right = right - indent0
            cur_bottom = bottom
            trims[1] = min(trims[1], image.shape[0] - bottom)
            trims[3] = min(trims[3], top)
            trims[0] = min(trims[0], left)
            trims[2] = min(trims[2], image.shape[1] - right)

            if indent_level is None:
                indent = int(left) - indent0
                indents[indent] = indents.get(indent, 0) + 1
                tabstops = [-89, 0, 89, 133, 177, 177 + (177 - 133)]
                for i in range(1, len(tabstops) - 1):
                    if (tabstops[i - 1] + tabstops[i]) / 2 <= indent < (tabstops[i] + tabstops[i + 1]) / 2:
                        indent_level = i - 1
                        break

            if bottom - top > 120 or prev_paragraph == "image" and indent_level is None:
                if prev_paragraph == "image":
                    top = rows[line - 1][1]
                left, right = chunks[0][0], chunks[-1][1]
                left = min(left, image.shape[1] - right)
                right = max(right, image.shape[1] - left)
                new_image[top:bottom, left:right] = image[top:bottom, left:right]

                if prev_paragraph == "quote":
                    write_file(r"\\end{quotation}")
                elif prev_paragraph == "chapter section":
                    write_file("}")
                prev_paragraph = "image"

                ocr_image = f"ocr-image-p{page_number}-{line}.png"
                width = str(round((right - left) / 1268, 2))
                write_file("\n\n\\begin{center}" +
                           "\n\\noindent\\includegraphics[width=" + width + "\\linewidth]{" + ocr_image + "}" +
                           "\n\\end{center}")
                print(f"    ocr image {right - left}x{bottom - top}+{left}+{top}")
                cv2.imwrite(ocr_image, image[top:bottom, left:right])
                break

            s = model.predict(input=image[top-5:bottom+10, left-5:right+10])[0]["rec_text"]
            if s.find("<") != -1 or s.find(">") != -1:
                t = s.replace("<", "〈").replace(">", "〉")
                print(f"    +replace {s} -> {t}")
                s = t
            if cur_paragraph == "chapter section":
                write_file(s)
            elif cur_paragraph == "引":
                if s == "言":
                    cur_paragraph = "引言"
                    write_file("\n\n\\section*{引言}\\addcontentsline{toc}{section}{引言}")
                else:
                    print(f"    引 without 言: {s}")
                    cur_paragraph = "paragraph"
                    if prev_paragraph != "paragraph" or prev_right is not None and prev_right < 1240:
                        write_file("\n\n\\noindent ")
                    write_file(s)
            elif cur_paragraph is not None:
                write_file(" " + s)
            elif indent_level == 0:
                if prev_paragraph == "quote":
                    write_file("\\end{quotation}")
                elif prev_paragraph == "chapter section":
                    write_file("}")
                if bottom - top > 49 and (prev_paragraph != "paragraph" or prev_right is None or prev_right < 1240) and len(s) >= 3 and s[0] == "第" and s[2] == "章":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\chapter{")
                elif bottom - top > 49 and (prev_paragraph != "paragraph" or prev_right is None or prev_right < 1240) and len(s) >= 3 and s[0] == "第" and s[2] == "節":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\section{")
                elif bottom - top > 49 and (prev_paragraph != "paragraph" or prev_right is None or prev_right < 1240) and s == "附錄":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\appendix\\chapter{")
                    appendix = True
                elif (prev_paragraph != "paragraph" or prev_right is None or prev_right < 1240) and s == "引":
                    cur_paragraph = "引"
                elif (prev_paragraph != "paragraph" or prev_right is None or prev_right < 1240) and len(s) >= 3 and s[0] == "第" and s[2] == "段":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\subsection{")
                elif appendix and (prev_paragraph != "paragraph" or prev_right is None or prev_right < 1240) and len(s) >= 2 and s[1] == "、" and s[0] in "一二三四五六七":
                    cur_paragraph = "附錄"
                    write_file("\n\n\\section*{" + s + "}\\addcontentsline{toc}{section}{" + s + "}")
                elif s == "《心體與性體》全集本編校說明":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\chapter{" + s)
                elif s == "序":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\chapter{" + s)
                else:
                    cur_paragraph = "paragraph"
                    if prev_paragraph != "paragraph" or prev_right is not None and prev_right < 1240:
                        write_file("\n\n\\noindent ")
                    write_file(s)
            elif indent_level == 1:
                if prev_paragraph == "quote" and prev_bottom is not None and top - prev_bottom < 50:
                    # Some noise would mistakenly decrease indent_level
                    cur_paragraph = "quote"
                    write_file("\n" + s)
                else:
                    if prev_paragraph == "quote":
                        write_file("\\end{quotation}")
                    elif prev_paragraph == "chapter section":
                        write_file("}")
                    cur_paragraph = "paragraph"
                    write_file("\n\n" + s)
            elif indent_level == 2 or indent_level == 3:
                if prev_paragraph == "paragraph" and prev_bottom is not None and top - prev_bottom < 50:
                    if prev_paragraph == "quote":
                        write_file("\\end{quotation}")
                    elif prev_paragraph == "chapter section":
                        write_file("}")
                    print(f"    ambiguous indent {top - prev_bottom} prev_right {prev_right}")
                    cur_paragraph = "paragraph"
                    if prev_right < 1240:
                        write_file("\n\n" + s)
                    else:
                        write_file("\n" + s)
                else:
                    cur_paragraph = "quote"
                    if prev_paragraph == "chapter section":
                        write_file("}")
                    if prev_paragraph != "quote":
                        write_file("\n\n\\begin{quotation}\\kaishu " + s)
                    elif prev_right is not None and prev_right < 1240:
                        write_file("\n\n" + s)
                        print(f"    quote newline")
                    else:
                        write_file("\n" + s)
            elif indent > 600 and right > 1240:
                if prev_paragraph == "quote":
                    write_file("\\end{quotation}")
                elif prev_paragraph == "chapter section":
                    write_file("}")
                cur_paragraph = "right"
                write_file("\n\\begin{flushright}" + s + "\\end{flushright}")
            else:
                print(f"    big indent")
                cur_paragraph = prev_paragraph
                write_file("\n" + s)
        if cur_paragraph is not None:
            prev_paragraph = cur_paragraph
        if cur_right is not None:
            prev_right = cur_right
        if cur_bottom is not None:
            prev_bottom = cur_bottom
    if prev_paragraph == "quote":
        write_file("\\end{quotation}")
    write_file("", flush=True)

    k = max(0, (image.shape[1] - trims[0] - trims[2]) * 4 / 3 - (image.shape[0] - trims[1] - trims[3]))
    k1 = k * trims[1] / (trims[1] + trims[3])
    k3 = k * trims[3] / (trims[1] + trims[3])
    trims[1] -= k1
    trims[3] -= k3
    trim = " ".join(map(lambda x: str(max(0, int(x*0.75) - 10))+"pt", trims))
    write_file("\n\n\\newpage\\thispagestyle{empty}\\addtocounter{page}{-1}\\vspace*{-12mm}\\begin{center}\\noindent\n\\includegraphics[clip, trim=" + trim + ", height=162mm]{" + image_path + "}\\end{center}", flush=True, crosscheck_only=True)
    return new_image


front_matters = [None,1,2,3,5,6,None,None,None,None,None]
page_number = 1
for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith((".png", ".jpg")):
        continue

    front_page_number = None
    if front_matters:
        if front_matters[0] is None:
            del front_matters[0]
            continue
        front_page_number = front_matters[0]
        del front_matters[0]

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {filename}")
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)
    if front_page_number is not None:
        # continue # debug
        if front_page_number == 5:
            write_file("\n\n\\newpage\\thispagestyle{empty}")
        new_image = ocr(image, front_page_number, f"({front_page_number})")
    elif page_number == 1:
        write_file(r"""

\newpage\markright{}

\tableofcontents

\mainmatter
\fancyhead[LE]{\small \thepage\ $\odot$\ \kaishu\leftmark}
\fancyhead[RO]{\small \kaishu\rightmark\ $\odot$\ \thepage}

\newpage

\markright{}

\part{綜論}""", flush=True)
        page_number += 1
        continue
    elif page_number == 335:
        write_file(r"""
\newpage

\markright{}

\part{分論一\newline 濂溪與橫渠}""", flush=True)
        page_number += 1
        continue
    elif 3 <= page_number <= 688: # debug
        new_image = ocr(image, page_number, str(page_number))
        page_number += 1
    else:
        page_number += 1
        continue

    # debug
    # cv2.imshow("cv", new_image)
    # if cv2.waitKey(0) == ord("q"):
    #     print("quit")
    #     break

write_file("\n\n\\end{document}\n\n", flush=True)

# cv2.destroyAllWindows()
print(sorted(list(indents.items())))


