from datetime import datetime
import sys
import os

import cv2
import numpy as np
import density
import mzs


text_width = 1270
input_dir = "牟宗三全集7 心體與性體（第三冊）/"
tex = density.TexWriter(os.path.join(input_dir, "ocr.tex"), os.path.join(input_dir, "crosscheck.tex"))
tex.write(mzs.preemble, "w")
tex.write(r"""\renewcommand{\chaptermark}[1]{\markboth{心體與性體 \quad 第三冊}{}}
\renewcommand{\sectionmark}[1]{\markboth{心體與性體 \quad 第三冊}{}}
\renewcommand{\subsectionmark}[1]{\markboth{心體與性體 \quad 第三冊}{}}

\begin{document}

\frontmatter

\begin{titlepage}
  \begin{flushright}
    \vspace*{\fill}
    {\heiti 牟宗三先生全集\textcircled{7}}

    \medskip

    {\Huge\songti 心體與性體}

    \smallskip

    {\kaishu （第三冊）}

    \smallskip

    {\songti 牟宗三 \quad 著}
    \vspace*{\fill}
  \end{flushright}
\end{titlepage}

\pagenumbering{arabic}
\fancyhead[LE]{\small (\thepage )\ $\odot$\ \kaishu\leftmark}
\fancyhead[RO]{\small \kaishu\rightmark\ $\odot$\ (\thepage )}

""")


scanned_images = [x for x in sorted(os.listdir(input_dir)) if x.endswith((".png", ".jpg"))]
pages = [("frontmatters", page, scanned_images.pop(0)) for page in ["cover",1,2,3,4]]
pages += [("mainmatters", page, scanned_images.pop(0)) for page in range(1, 619)]
assert len(scanned_images) == 4, f"scanned_images: {scanned_images}"
tabstopper = density.Tabstopper(
    [-89, 0, 89, 135, 180, 203, text_width],
    dict([("(", 14), ("「", 13), ("〔〕", 18), ("《", 14), ("〈", 19)]),
    dict([("，", 16), ("、", 13), ("。", 12), ("》", 15), ("〈", 15), ("〕", 15), ("」", 15), ("；", 15), ("：", 15)])
)
ocr_cache = density.OCRCache("牟宗三全集7 心體與性體（第三冊）.txt")


prev_page_state = None
for matter, page_number, filename in pages:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {page_number} {filename}")
    if filename is None:
        print(f"<missing page>")
        with open(tex.crosscheck, "a") as f:
            f.write(r"\newpage\thispagestyle{empty}\addtocounter{page}{-1}")
        continue
    if (matter, page_number) == ("frontmatters", "cover"):
        continue
    if (matter, page_number) == ("frontmatters", 1):
        tex.write(r"""\newpage\markright{}

\tableofcontents

\mainmatter
\fancyhead[LE]{\small \thepage\ $\odot$\ \kaishu\leftmark}
\fancyhead[RO]{\small \kaishu\rightmark\ $\odot$\ \thepage}

""")
        continue
    if matter == "frontmatters" and 2 <= page_number <= 4:
        continue
    if (matter, page_number) == ("mainmatters", 1):
        tex.writeln(r"\setcounter{part}{3}")
        tex.writeln(r"\part{分論三 \quad 朱子}\setcounter{chapter}{0}")
        tex.writeln()
        continue
    if (matter, page_number) == ("mainmatters", 2):
        continue
    if matter == "mainmatters" and page_number in [196, 392, 496, 540]:
        with open(tex.crosscheck, "a") as f:
            f.write(r"\newpage\thispagestyle{empty}\addtocounter{page}{-1}")
        continue
    if len(sys.argv) > 1:
        if matter == "frontmatters" or page_number < int(sys.argv[1]):
            continue
    image = cv2.imread(os.path.join(input_dir, filename))
    if matter == "mainmatters":
        if page_number == 230:
            image[793:878, 316:358, :] = 255
        elif page_number == 277:
            image[:100, :, :] = 255
        elif page_number == 278:
            image[:, -50:, :] = 255
        elif page_number == 328:
            image[1036:1111, 337:364, :] = 255
        elif page_number == 545:
            image[573:648, 314:339, :] = 255
    image_black = np.mean(1 - image / 255, axis=2)
    try:
        plugins = {}
        if matter == "mainmatters":
            if page_number == 159:
                plugins["further trim"] = 0.01
            elif page_number in [381, 414]:
                plugins["close quotation"] = True
            elif page_number == 451:
                plugins["close chapter"] = True
            if page_number >= 241:
                plugins["tab 2 quotation itemize"] = True

        rows, output_image = density.ocr(filename, image / 255, ocr_cache, plugins=plugins)
        media_rows = [("text", bound, box_texts) for bound, box_texts in rows]
        def make_figure(l, h):
            bound = media_rows[l][1]
            for _, b, _ in media_rows[l:h]:
                bound = density.boxes_union(bound, b)
            rl, rh, cl, ch = bound
            trim = " ".join(f"{x/96-0.1:.2f}in" for x in [cl, image_black.shape[0] - rh, image_black.shape[1] - ch, rl])
            size = f"width={(ch - cl) / text_width:.2f}\\textwidth"
            media_rows[l:h] = [("figure", bound, (trim, size, filename))]
            cv2.imwrite(f"figure-{page_number}.png", image[rl:rh, cl:ch])
        if (matter, page_number) == ("mainmatters", 322):
            make_figure(11, 19)
        elif (matter, page_number) == ("mainmatters", 323):
            make_figure(2, 13)
        left_margin = mzs.write_header(tex, matter, page_number, media_rows[0], text_width)
        prev_page_state = mzs.write_page(tex, matter, page_number, media_rows[1:], left_margin, tabstopper, image_black, prev_page_state, plugins)

        page_bound = media_rows[0][1]
        for media, bound, _ in media_rows[1:]:
            if media != "figure":
                page_bound = density.boxes_union(page_bound, bound)
        rl, rh, cl, ch = page_bound
        trim = " ".join(f"{x/96-0.1:.2f}in" for x in [cl, image_black.shape[0] - rh, image_black.shape[1] - ch, rl])
        if (ch - cl) / (rh - rl) * 162 < 122:
            size = "height=162mm"
        else:
            size = "width=122mm"
        with open(tex.crosscheck, "a") as f:
            f.write(r"""\newpage\thispagestyle{empty}\addtocounter{page}{-1}\vspace*{-12mm}
\begin{center}\noindent
\includegraphics[clip, trim=""" + trim + ", " + size + "]{" + filename + r"""}
\end{center}

""")
    except Exception as e:
        print(e)
        density.ocr(filename, image / 255, ocr_cache, verbose=True, predict=False)
        raise
tex.writeln(r"\end{document}")
tex.writeln()


