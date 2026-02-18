from datetime import datetime
import os
import cv2
import numpy as np
from paddleocr import TextRecognition


input_dir = "ocr-input/"
markdown_file = "ocr.md"
crosscheck_markdown_file = "crosscheck-ocr.md"


buf = []
def write_file(s, flush=False, crosscheck_only=False, mode="a"):
    global buf
    if not flush:
        buf.append(s)
        return
    s = "".join(buf) + s
    buf = []
    if not crosscheck_only:
        with open(markdown_file, mode) as file:
            file.write(s)
    with open(crosscheck_markdown_file, mode) as file:
        file.write(s)


write_file(r"""
---
title: "心體與性體第一冊"
author: "牟宗三"
documentclass: book
geometry:
  - top=10mm
  - left=2mm
  - bottom=0mm
  - right=2mm
  - headsep=2mm
  - paperwidth=108mm
  - paperheight=170mm
CJKmainfont: "Chiron Sung HK"
CJKsansfont: "DFHKStdKai-B5"
fontsize: 12pt
header-includes:
  - \usepackage{setspace}
  - \setstretch{1.25}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\footnotesize\textsf\leftmark}
  - \fancyhead[R]{\footnotesize\textsf\rightmark}
  - \fancypagestyle{plain}{
      \fancyhf{}
      \fancyhead[L]{\footnotesize\textsf\leftmark}
      \fancyhead[R]{\footnotesize\textsf\rightmark}
    }
  - \renewcommand{\headrulewidth}{0pt}
  - \usepackage{indentfirst}
  - \setlength{\parskip}{0pt}
  - \renewenvironment{quote} {\list{}{\leftmargin=10mm \rightmargin=0em}\item\relax} {\endlist}
  - \usepackage{graphicx}
  - \renewcommand{\chaptermark}[1]{\markboth{}{}}
---
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


model = TextRecognition()
page_number = -10
indents = {}
manual_crop = {
    261: (0, -50),
    263: (100, -1),
    621: (50, -1)
}
for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith((".png", ".jpg")):
        continue
    if not 3 <= page_number <= 688:
        page_number += 1
        continue
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {filename}")
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)
    new_image = image.copy()
    new_image[:, :] = (255, 100, 0)

    # remove left right glitches
    for (left, right) in black_white_chunks(np.where(np.all(image == 255, axis=(0,2)), 1, 0)):
        if right - left < 5:
            print(f"    left right glitch {left}+{right - left}")
            new_image[:, left:right] = image[:, left:right]
            image[:, left:right] = (255, 255, 255)
    if page_number in manual_crop:
        print(f"    manual crop {manual_crop[page_number]}")
        new_image[:, :manual_crop[page_number][0]] = image[:, :manual_crop[page_number][0]]
        new_image[:, manual_crop[page_number][1]:] = image[:, manual_crop[page_number][1]:]
        image[:, :manual_crop[page_number][0]] = (255, 255, 255)
        image[:, manual_crop[page_number][1]:] = (255, 255, 255)

    trims = [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
    rows = black_white_chunks(np.where(np.all(image == 255, axis=(1,2)), 1, 0))

    # page header
    while rows:
        (top, bottom) = rows[0]
        headers = black_white_chunks(np.where(np.all(image[top:bottom] == 255, axis=(0,2)), 1, 0), link=11)

        for i, (left, right) in enumerate(headers):
            new_image[top:bottom, left:right] = image[top:bottom, left:right]
            # if i > 0:
            #     print(f"    header gap {left - headers[i - 1][1]}")
            # print(f"    header chunk {right - left}")

        def predict(i, j=None):
            if j is None:
                j = i
            if i < 0:
                i += len(headers)
            if j < 0:
                j += len(headers)
            if 0 <= i <= j <= len(headers):
                return model.predict(input=image[top:bottom, headers[i][0]:headers[j][1]])[0]["rec_text"]
            return None

        if bottom - top > 10 and headers[-1][1] - headers[0][0] > 10:
            if page_number % 2 == 0 and predict(0) == str(page_number):
                trims[1] = min(trims[1], image.shape[0] - bottom)
                trims[3] = min(trims[3], top)
                trims[0] = min(trims[0], headers[0][0])
                trims[2] = min(trims[2], image.shape[1] - headers[-1][1])
                indent0 = int(headers[0][0])
                # odot = predict(1)
                book = predict(2) if len(headers) >= 3 else "-"
                volume = predict(3) if len(headers) >= 4 else "-"
                write_file("\n\n\\newpage\\markboth{" +
                           f"{page_number} $\\odot$ {book} {volume}" +
                           "}{}")
                rows = rows[1:]
                break
            elif page_number % 2 == 1 and predict(-1) == str(page_number):
                trims[1] = min(trims[1], image.shape[0] - bottom)
                trims[3] = min(trims[3], top)
                trims[0] = min(trims[0], headers[0][0])
                trims[2] = min(trims[2], image.shape[1] - headers[-1][1])
                indent0 = int(headers[-1][1]) - 1268
                section = predict(0) if len(headers) >= 5 else "-"
                chapter = predict(1) if len(headers) >= 4 else "-"
                title = predict(2, -3) if len(headers) >= 3 else "-"
                # odot = predict(-2)
                write_file("\n\n\\newpage\\markboth{}{" +
                           f"{section} {chapter} {title} $\\odot$ {page_number}" +
                           "}")
                rows = rows[1:]
                break
        print(f"    noisy header {right - left}x{bottom - top}+{left}+{top}")
        rows = rows[1:]

    # page content
    prev_paragraph = None
    prev_right = None
    prev_bottom = None
    for line, (top, bottom) in enumerate(rows):
        chunks = black_white_chunks(np.where(np.all(image[top:bottom] == 255, axis=(0,2)), 1, 0), link=50)

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
                    write_file(r"}\\end{spacing}")
                elif prev_paragraph == "chapter section":
                    write_file("}")
                prev_paragraph = "image"

                ocr_image = f"ocr-image-p{page_number}-{line}.png"
                width = str(round((right - left) / 1268, 2))
                write_file("\n\n\\begin{center}" +
                           "\n\\includegraphics[width=" + width + "\\linewidth]{" + ocr_image + "}" +
                           "\n\\end{center}")
                print(f"    ocr image {right - left}x{bottom - top}+{left}+{top}")
                cv2.imwrite(ocr_image, image[top:bottom, left:right])
                break

            s = model.predict(input=image[top:bottom, left:right])[0]["rec_text"]
            if cur_paragraph is not None:
                write_file(f" {s}")
            elif indent_level == 0:
                if prev_paragraph == "quote":
                    write_file("}\\end{spacing}")
                elif prev_paragraph == "chapter section":
                    write_file("}")
                if len(s) >= 3 and s[0] == "第" and s[2] == "章":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\chapter{" + s)
                elif len(s) >= 3 and s[0] == "第" and s[2] == "節":
                    cur_paragraph = "chapter section"
                    write_file("\n\n\\section{" + s)
                else:
                    cur_paragraph = "paragraph"
                    if prev_paragraph != "paragraph" or prev_right is not None and prev_right < 1240:
                        write_file("\n\n")
                    write_file(s)
            elif indent_level == 1:
                if prev_paragraph == "quote" and prev_bottom is not None and top - prev_bottom < 50:
                    cur_paragraph = "quote"
                    write_file(s)
                else:
                    if prev_paragraph == "quote":
                        write_file("}\\end{spacing}")
                    elif prev_paragraph == "chapter section":
                        write_file("}")
                    cur_paragraph = "paragraph"
                    write_file(f"\n\n　　{s}")
            elif indent_level == 2 or indent_level == 3:
                if prev_paragraph == "paragraph" and prev_bottom is not None and top - prev_bottom < 50:
                    print(f"    ambiguous indent {top - prev_bottom} prev_right {prev_right}")
                    cur_paragraph = "paragraph"
                    if prev_right < 1240:
                        write_file("\n\n　　　")
                    write_file(s)
                else:
                    cur_paragraph = "quote"
                    if prev_paragraph != "quote":
                        if prev_paragraph is not None:
                            write_file("\n\n\\vspace{1em}")
                        write_file("\n\n> \\begin{spacing}{1.5}\\footnotesize\\textsf{")
                    elif prev_right is not None and prev_right < 1240:
                        write_file("\\newline\n")
                        print(f"    quote newline")
                    write_file(s)
            elif indent_level == 3:
                cur_paragraph = prev_paragraph
                write_file(s)
            else:
                cur_paragraph = prev_paragraph
                write_file(s)
        if cur_paragraph is not None:
            prev_paragraph = cur_paragraph
        if cur_right is not None:
            prev_right = cur_right
        if cur_bottom is not None:
            prev_bottom = cur_bottom
    if prev_paragraph == "quote":
        write_file("}\\end{spacing}")
    write_file("", flush=True)

    trim = " ".join(map(lambda x: str(int(x*0.75))+"pt", trims))
    write_file("\n\n\\newpage\\markboth{}{}\n\\includegraphics[clip, trim=" + trim + ", width=\\linewidth]{" + image_path + "}", flush=True, crosscheck_only=True)

    # cv2.imshow("cv", new_image)
    # if cv2.waitKey(0) == ord("q"):
    #     print("quit")
    #     break
    page_number += 1

# cv2.destroyAllWindows()
print(sorted(list(indents.items())))


