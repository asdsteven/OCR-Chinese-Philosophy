import cv2
import numpy as np
from paddleocr import TextRecognition


model = TextRecognition()


def nonzero_chunks(a):
    # a is 1D boolean array
    boundaries = np.where(np.diff(a) != 0)[0]
    if boundaries.size == 0:
        if a.size > 0 and a[0]:
            return [(0, len(a))]
        return []
    chunks = [(a[0], 0, boundaries[0] + 1)]
    for i, x in enumerate(boundaries[:-1]):
        chunks.append((a[x + 1], x + 1, boundaries[i + 1] + 1))
    chunks.append((a[-1], boundaries[-1] + 1, len(a)))
    return [(l,h) for x, l, h in chunks if x]


def boxes_union(box1, box2):
    rl, rh, cl, ch = box1
    rl2, rh2, cl2, ch2 = box2
    return (min(rl, rl2), max(rh, rh2), min(cl, cl2), max(ch, ch2))


def box_size(box):
    return min(box[1] - box[0], box[3] - box[2])


def trim_box(image_black, box, thresh=0):
    rl, rh, cl, ch = box
    r, c = np.nonzero(image_black[rl:rh, cl:ch])
    if not r.size or not c.size:
        print(f"<empty trim box {box}>")
        return rl, rh, cl, ch
    rl, rh = rl + r.min(), rl + r.max() + 1
    cl, ch = cl + c.min(), cl + c.max() + 1
    while True:
        if rl < rh and np.mean(image_black[rl, cl:ch]) <= thresh:
            rl += 1
        elif rl < rh and np.mean(image_black[rh-1, cl:ch]) <= thresh:
            rh -= 1
        elif cl < ch and np.mean(image_black[rl:rh, cl]) <= thresh:
            cl += 1
        elif cl < ch and np.mean(image_black[rl:rh, ch-1]) <= thresh:
            ch -= 1
        else:
            break
    return rl, rh, cl, ch


def expand_box(image_black, box, thresh=0.0001):
    rl, rh, cl, ch = box
    for i in range(5):
        if rl > 0 and np.all(image_black[rl-1, cl:ch] < thresh):
            rl -= 1
        if cl > 0 and np.all(image_black[rl:rh, cl-1] < thresh):
            cl -= 1
        if rh < image_black.shape[0] and np.all(image_black[rh, cl:ch] < thresh):
            rh += 1
        if ch < image_black.shape[1] and  np.all(image_black[rl:rh, ch] < thresh):
            ch += 1
    return (rl, rh, cl, ch)


colors = []
for b in range(0, 5):
    for g in range(0, 5):
        for r in range(0, 5):
            if b + g + r >= 4:
                colors.append([b / 4, g / 4, r / 4])
colors = np.array(colors)
np.random.shuffle(colors)


def render_boxes(canvas, image, boxes, tall_boxes=[], noises=[]):
    for i, (rl, rh, cl, ch) in enumerate(boxes):
        canvas[rl:rh, cl:ch] = image[rl:rh, cl:ch] * 0.7 + colors[i % len(colors)] * 0.3
    for rl, rh, cl, ch in noises:
        mrl, mrh = min(rl,rh-5), max(rh,rl+5)
        mcl, mch = min(cl,ch-5), max(ch,cl+5)
        canvas[mrl:mrh, mcl:mch] = image[mrl:mrh, mcl:mch] * 0.3 + np.array([0, 0, 1]) * 0.7
    for rl, rh, cl, ch in tall_boxes:
        canvas[rl:rh, cl:ch] = image[rl:rh, cl:ch] * 0.3 + np.array([1, 0, 0]) * 0.7


def wait_right_key():
    key = cv2.waitKey(0)
    if key == ord("q"):
        print("quit")
        quit()
    elif key == 2: # left arrow
        return -1
    elif key == 3: # right arrow
        return 1
    else:
        print(f"unknown key code: {key}")
        return 1


def square_densities(image_black, size=31):
    h = size // 2
    l = size // 2 + 1
    cumsum = np.pad(image_black, ((l, h), (l, h)), "constant", constant_values=0).cumsum(axis=0).cumsum(axis=1)
    squares = cumsum[size:, size:].copy()
    squares -= cumsum[:-size, size:]
    squares -= cumsum[size:, :-size]
    squares += cumsum[:-size, :-size]
    squares /= size**2
    return squares


def row_densities(squares, size=50, thresh=0.001):
    rows = []
    for row in squares:
        s, n = 0, 0
        chunks = nonzero_chunks(row > thresh)
        for l, r in chunks:
            # if r - l < size:
                # continue
            s += np.sum(row[l:r])
            n += r - l
        rows.append(s / n if n > 0 else 0)
    return rows


def d_densities(rows, size=15, thresh=0.07):
    drows = [[0, 0, 0]] * (size // 2)
    for l in range(len(rows) - size + 1):
        r = l + size - 1
        d = rows[r] - rows[l]
        if d > thresh:
            drows.append([0, min(1, 0.5 + d), 0])
        elif d < -thresh:
            drows.append([0, 0, min(1, 0.5 - d)])
        else:
            drows.append([0, 0, 0])
    drows += [[0, 0, 0]] * (size // 2)
    return drows


def window_center(rows, l, r):
    l_sum, l_weight = 0, 0
    r_sum, r_weight = 0, 0
    n = r - l
    for i, x in enumerate(rows[l+1:r+1]):
        r_sum += x
        r_weight += x * (n - i)
    i = l
    min_diff, min_i = r_weight, i
    while i <= r:
        l_weight -= l_sum
        l_weight += rows[i] * n
        l_sum += rows[i]
        i += 1
        r_sum -= rows[i]
        r_weight -= rows[i] * n
        r_weight += r_sum
        diff = abs(l_weight - r_weight)
        if diff < min_diff:
            min_diff, min_i = diff, i
    return min_i


def peak_troughs(rows, size=15, thresh=0.07):
    pts = []
    state = None
    earliest_up, earliest_down = None, None
    furthest_up, furthest_down = None, None
    for l in range(len(rows) - size + 1):
        r = l + size - 1
        d = rows[r] - rows[l]
        if d > thresh:
            furthest_up = r
            if state != "up":
                if earliest_up is not None:
                    pts.append(("peak-before", earliest_up))
                    pts.append(("peak", window_center(rows, earliest_up, furthest_down)))
                    pts.append(("peak-after", furthest_down))
                earliest_up = l
            state = "up"
        elif d < -thresh:
            furthest_down = r
            if state != "down":
                if earliest_down is not None:
                    pts.append(("trough-before", earliest_down))
                    pts.append(("trough", window_center([1 - x for x in rows], earliest_down, furthest_up)))
                    pts.append(("trough-after", furthest_up))
                earliest_down = l
            state = "down"
    if earliest_up is not None and furthest_down is not None and furthest_down > earliest_up:
        pts.append(("peak-before", earliest_up))
        pts.append(("peak", window_center(rows, earliest_up, furthest_down)))
        pts.append(("peak-after", furthest_down))
    return pts


def link_col_boxes(boxes, size=60):
    i = 0
    while i + 1 < len(boxes):
        if boxes[i + 1][2] - boxes[i][3] < size:
            boxes[i] = boxes_union(boxes[i], boxes[i + 1])
            del boxes[i + 1]
        else:
            i += 1
    return boxes


class OCRCache:
    def __init__(self, cache_filename):
        self.filename = cache_filename
        self.cache = {}
        try:
            with open(self.filename, "r") as f:
                for line in f:
                    filename, rl, rh, cl, ch, text = line.split(",",maxsplit=5)
                    self.cache[filename, int(rl), int(rh), int(cl), int(ch)] = text.strip()
        except FileNotFoundError:
            print("no ocr cache")

    def write(self, filename, rl, rh, cl, ch, text):
        self.cache[filename, rl, rh, cl, ch] = text
        with open(self.filename, "a") as f:
            f.write(f"{filename},{rl},{rh},{cl},{ch},{text}\n")


def ocr(filename, image, ocr_cache, verbose=False, predict=True):
    print("ocr row densities")
    image_black = np.mean(1 - image, axis=2)

    squares = square_densities(image_black)
    if verbose:
        cv2.imshow("cv", image * squares[:, :, None])
        wait_right_key()

    rows = row_densities(squares)
    if verbose:
        cv2.imshow("cv", image * np.array(rows)[:, None, None])
        wait_right_key()

    drows = d_densities(rows)
    if verbose:
        cv2.imshow("cv", image * np.array(drows)[:, None, :])
        wait_right_key()

    pts = peak_troughs(rows, thresh=0.07)
    if not pts:
        print(f"<blank page>")
        return None, None
    troughs = [pts[0][1]] + [t for s, t in pts if s == "trough"] + [pts[-1][1]]
    row_boxes = [trim_box(image_black, (t, b, 0, image.shape[1])) for t, b in zip(troughs, troughs[1:])]
    # row_boxes = []
    # for i, (s, t) in enumerate(pts):
    #     if s == "peak":
    #         row_boxes.append(trim_box(image_black, (pts[i - 1][1], pts[i + 1][1], 0, image.shape[1] - 1)))
    canvas = np.zeros_like(image)
    render_boxes(canvas, image, row_boxes)
    if verbose:
        cv2.imshow("cv", canvas)
        wait_right_key()

    print("ocr col densities")
    col_densities = np.zeros_like(image_black)
    col_canvas = np.zeros_like(image)
    col_boxes = []
    for rl, rh, cl, ch in row_boxes:
        cols = row_densities(squares[rl:rh, :].transpose(), size=20)
        col_densities[rl:rh, :] = np.array(cols)[None, :]

        dcols = d_densities(cols, size=15, thresh=0.01)
        col_canvas[rl:rh, :] = np.array(dcols)[None, :, :]

        boxes = []
        pts = peak_troughs(cols, size=15, thresh=0.01)
        for i, (s, t) in enumerate(pts):
            if s == "peak":
                boxes.append(trim_box(image_black, (rl, rh, pts[i - 1][1], pts[i + 1][1])))
                # boxes.append((rl, rh, pts[i - 1][1], pts[i + 1][1]))
        col_boxes.append(link_col_boxes(boxes))

    if verbose:
        cv2.imshow("cv", canvas * 0.3 + canvas * 0.7 * col_densities[:, :, None])
        wait_right_key()

    if verbose:
        cv2.imshow("cv", canvas * 0.3 + col_canvas * 0.7)
        wait_right_key()

    canvas2 = np.zeros_like(image)
    for boxes in col_boxes:
        tight_boxes = []
        for box in boxes:
            tight_boxes.append(expand_box(image_black, box))
            tight_box = trim_box(image_black, box, 0.01)
            if tight_box != box:
                tight_boxes.append(tight_box)
        render_boxes(canvas2, image, tight_boxes)
    output_image = image * 0.5 + canvas2 * 0.5

    if verbose:
        cv2.imshow("cv", output_image)
        wait_right_key()

    print("ocr box predict")
    row_texts = []
    for boxes in col_boxes:
        bound = None
        box_texts = []
        for rl, rh, cl, ch in boxes:
            if ch - cl < 10 and rh - rl < 10:
                continue
            b = expand_box(image, (rl, rh, cl, ch))
            contrast = (image[b[0]:b[1], b[2]:b[3]] * 255).round().astype(np.uint8)
            if predict:
                text = ocr_cache.cache.get((filename, b[0], b[1], b[2], b[3]), None)
                if text is None:
                    text = model.predict(input=contrast)[0]["rec_text"]
                    ocr_cache.write(filename, b[0], b[1], b[2], b[3], text)
            else:
                text = ""
            box_texts.append(((rl, rh, cl, ch), text.replace("（", "(").replace("）", ")")))
            if bound is None:
                bound = (rl, rh, cl, ch)
            else:
                bound = boxes_union(bound, (rl, rh, cl, ch))
        indent = "" if bound is None else " " * max(0, round(bound[2] / 30 - 10))
        top, left, right = (-1, -1, -1) if bound is None else (bound[0], bound[2], bound[3])
        print(f"    {top:4d} {left:4d} {right:4d} {indent} {"    ".join(t for _, t in box_texts)}")
        row_texts.append((bound, box_texts))

    return row_texts, output_image


tex_symbols = {
    ord("Ⅱ"): "II",
    ord("Ⅲ"): "III",
    ord("Ⅳ"): "IV",
    ord("・"): r"$\cdot$",
    ord("⊙"): r"$\odot$"
}


class TexWriter:
    def __init__(self, ocr, crosscheck):
        self.ocr = ocr
        self.crosscheck = crosscheck

    def write(self, s, mode="a"):
        t = s.translate(tex_symbols)
        with open(self.ocr, mode) as f:
            f.write(t)
        with open(self.crosscheck, mode) as f:
            f.write(t)

    def writeln(self, s=""):
        self.write(s + "\n")


class Tabstopper:
    def __init__(self, tabstops, left_compensates, right_compensates):
        self.tabstops = tabstops
        self.left_compensates = left_compensates
        self.right_compensates = right_compensates

    def tab(self, indent):
        for i in range(0, len(self.tabstops) - 1):
            if indent < (self.tabstops[i] + self.tabstops[i + 1]) / 2:
                return i
        return len(self.tabstops) - 1

    def indent_tab(self, left_margin, box_texts):
        for box, text in box_texts:
            if text:
                indent = box[2] - self.left_compensates.get(text[0], 0) - left_margin
                if self.tab(indent) > 0:
                    break
        else:
            return None, None, None, (None, None)
        for box, text in reversed(box_texts):
            if text:
                right_indent = box[3] + self.right_compensates.get(text[-1], 0) - left_margin
                return indent, self.tab(indent), self.tabstops[-1] - right_indent, (box, text)

    def normalize_left_margin(self, left_margin, rows):
        lefts = []
        for box_texts in rows:
            indent, tab, _, (box, _) = self.indent_tab(left_margin, box_texts)
            if tab is not None and 1 <= tab <= 3:
                lefts.append(box[2] - self.tabstops[tab])
        if len(lefts) < 3:
            return left_margin
        lefts.sort()
        n = len(lefts) // 3
        return round(sum(lefts[n:-n]) / (len(lefts) - n - n))

    def tab_outliers(self, left_margin, rows):
        for box_texts in rows:
            indent, tab, _, (_, text) = self.indent_tab(left_margin, box_texts)
            if tab is not None and 1 <= tab <= 3:
                yield abs(indent - self.tabstops[tab]), text


一至十 = "一二三四五六七八九十"


def split_number(s, patterns=["1234567890", "一二三四五六七八九十"], noise=".、()"):
    for pattern in patterns:
        i = 0
        number = False
        while i < len(s):
            if s[i] in pattern:
                number = True
                i += 1
            elif s[i] in noise:
                i += 1
            else:
                break
        if number:
            return s[:i], s[i:]
    return "", s


def match_chapter(sme, start, end):
    if not sme.startswith(start):
        return False
    me = sme.removeprefix(start)
    if not me.endswith(end):
        return False
    m = me.removesuffix(end)
    if not m:
        return False
    if not all(c in 一至十 for c in m):
        return False
    return True


def split_chapter(s, end=("部", "章", "節", "段")):
    if s.startswith("第"):
        number, content = split_number(s[1:], 一至十)
        if number and content.startswith(end):
            return s[0] + number + content[0], content[1:].strip()
    elif s.startswith("分論"):
        number, content = split_number(s[2:], 一至十)
        if number:
            return s[:2] + number, content.strip()
    elif s.startswith("附錄"):
        return s[:2], s[2:].strip()
    return "", s


def normalize_header_mark(mark):
    chapters = []
    while True:
        chapter, content = split_chapter(mark)
        if not chapter:
            break
        chapters.append(chapter)
        mark = content
    return r" \quad ".join(chapters) + r" \quad " + mark


def tight_bound(image_black, box_texts, thresh):
    boxes = [box for box, text in box_texts if text]
    bound = trim_box(image_black, boxes[0], thresh)
    for box in boxes[1:]:
        bound = boxes_union(bound, trim_box(image_black, box, thresh))
    return bound


