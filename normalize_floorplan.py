"""
RoomView Floor Plan Normalization Tool

Converts floor plan files (PNG, JPG, SVG, PDF) into clean,
drafting-style SVG with standardized wall thickness.

Pipeline for raster inputs (PNG/JPG):
    1. Convert to grayscale
    2. Binary threshold
    3. Remove small connected components
    4. Morphological close
    5. Skeletonize walls to centerlines
    6. Detect outer vs interior walls
    7. Standardize wall thickness (dilate to target widths)
    8. Potrace vector trace (fill mode)
    9. Remove transforms, style SVG

For SVG inputs:
    - Normalize styles only (no Potrace)

For PDF inputs:
    - Convert to SVG via pdf2svg
    - Normalize styles only

Usage:
    python normalize_floorplan.py                    (batch: all images in input/)
    python normalize_floorplan.py floor3.png          (single file)
    python normalize_floorplan.py plan.pdf             (PDF input)
    python normalize_floorplan.py drawing.svg          (SVG input)
"""

import argparse
import base64
import logging
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import uuid
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger("roomview")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_POTRACE = os.path.join(SCRIPT_DIR, "potrace", "potrace-1.16.win64", "potrace.exe")
POTRACE_BIN = os.path.abspath(_LOCAL_POTRACE) if os.path.exists(_LOCAL_POTRACE) else "potrace"
PDF2SVG_BIN = "pdf2svg"
SCRATCH_ROOT = os.path.join(SCRIPT_DIR, "output", "_tmp")

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

SUPPORTED_RASTER = (".png", ".jpg", ".jpeg")
SUPPORTED_VECTOR = (".svg",)
SUPPORTED_PDF = (".pdf",)


def _make_work_dir(root, prefix):
    """
    Create a per-run scratch directory under root.

    Avoid tempfile.mkdtemp() here: in some locked-down Windows environments it can
    create directories that are not writable, which breaks OpenCV/Potrace I/O.
    """
    os.makedirs(root, exist_ok=True)
    for _ in range(100):
        path = os.path.join(root, f"{prefix}{uuid.uuid4().hex[:12]}")
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"Could not create work dir under: {root}")


# ---------------------------------------------------------------------------
# Stroke-based pipeline (no Potrace)
# ---------------------------------------------------------------------------

XLINK_NS = "http://www.w3.org/1999/xlink"


def _ensure_odd(value, minimum=3):
    v = max(int(value), int(minimum))
    return v if v % 2 == 1 else v + 1


def _debug_path(output_svg_path, suffix):
    base, _ = os.path.splitext(str(output_svg_path))
    return base + suffix


def _write_png(path, img):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _local_name(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _composite_alpha_on_white(img):
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, None]
        out = bgr * alpha + 255.0 * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return img


def load_raster_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img = _composite_alpha_on_white(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def render_pdf_first_page(pdf_path, dpi):
    """
    Rasterize the first page of a PDF to BGR using PyMuPDF (fitz).
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PDF support requires PyMuPDF (pip install PyMuPDF).") from exc

    doc = fitz.open(str(pdf_path))
    if doc.page_count < 1:
        raise RuntimeError("PDF has no pages.")

    page = doc.load_page(0)
    zoom = float(dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    n = pix.n
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, n))
    if n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    raise RuntimeError(f"Unexpected PDF pixmap channels: {n}")


def svg_contains_embedded_raster(tree):
    root = tree.getroot()
    for elem in root.iter():
        if _local_name(elem.tag) == "image":
            return True
    return False


def _decode_data_uri_image(data_uri):
    if not (isinstance(data_uri, str) and data_uri.startswith("data:") and "base64," in data_uri):
        return None
    b64 = data_uri.split("base64,", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def extract_embedded_svg_image(svg_path, tree):
    root = tree.getroot()
    svg_dir = os.path.dirname(os.path.abspath(str(svg_path)))
    for elem in root.iter():
        if _local_name(elem.tag) != "image":
            continue
        href = elem.get(f"{{{XLINK_NS}}}href") or elem.get("href")
        if not href:
            continue

        img = _decode_data_uri_image(href)
        if img is not None:
            img = _composite_alpha_on_white(img)
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.ndim == 3 and img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.ndim == 3 and img.shape[2] == 3:
                return img

        candidate = os.path.join(svg_dir, href)
        if os.path.exists(candidate):
            return load_raster_image(candidate)

    return None


def boost_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_binarize(gray, blocksize, C):
    bs = _ensure_odd(blocksize, minimum=3)
    # Walls are darker -> invert so walls become white (255)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        int(C),
    )


def morph_close(binary_walls, kernel_size):
    k = max(int(kernel_size), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(binary_walls, cv2.MORPH_CLOSE, kernel, iterations=1)


def remove_small_components_fg(binary_walls, min_area):
    # binary_walls: walls=255, background=0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_walls, connectivity=8)
    cleaned = np.zeros_like(binary_walls)
    kept = 0
    removed = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            cleaned[labels == label_id] = 255
            kept += 1
        else:
            removed += 1
    log.info(f"  Components kept: {kept}, removed: {removed} (area < {min_area})")
    return cleaned


def zhang_suen_thinning(binary_walls):
    """
    Zhang-Suen thinning.
    Input: binary walls mask (walls=255, background=0)
    Output: skeleton mask (skeleton=255, background=0)
    """
    img = (binary_walls > 0).astype(np.uint8)
    if img.size == 0:
        return binary_walls.copy()

    changed = True
    while changed:
        changed = False
        padded = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0)

        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )

        m1 = (img == 1)
        m2 = (neighbors >= 2) & (neighbors <= 6)
        m3 = (a == 1)
        m4 = (p2 * p4 * p6 == 0)
        m5 = (p4 * p6 * p8 == 0)
        to_delete = m1 & m2 & m3 & m4 & m5
        if np.any(to_delete):
            img[to_delete] = 0
            changed = True

        padded = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )

        m1 = (img == 1)
        m2 = (neighbors >= 2) & (neighbors <= 6)
        m3 = (a == 1)
        m4 = (p2 * p4 * p8 == 0)
        m5 = (p2 * p6 * p8 == 0)
        to_delete = m1 & m2 & m3 & m4 & m5
        if np.any(to_delete):
            img[to_delete] = 0
            changed = True

    return (img * 255).astype(np.uint8)


_OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def skeleton_to_polylines(skeleton):
    """
    Convert a 1px skeleton (255/0) into polylines.
    Returns list of polylines; each polyline is list of (y, x) ints.
    """
    ys_xs = np.argwhere(skeleton > 0)
    if ys_xs.size == 0:
        return []

    coords = [tuple(map(int, p)) for p in ys_xs]
    skel_set = set(coords)

    neighbor_map = {}
    for y, x in coords:
        neigh = []
        for dy, dx in _OFFSETS_8:
            p = (y + dy, x + dx)
            if p in skel_set:
                neigh.append(p)
        neighbor_map[(y, x)] = neigh

    nodes = {p for p, neigh in neighbor_map.items() if len(neigh) != 2}
    visited_edges = set()
    polylines = []

    def trace_from(start, first):
        points = [start]
        prev = start
        curr = first
        while True:
            points.append(curr)
            visited_edges.add(_edge_key(prev, curr))

            if curr in nodes and curr != start:
                break

            neigh = neighbor_map.get(curr, [])
            next_candidates = [p for p in neigh if p != prev]
            if not next_candidates:
                break
            nxt = next_candidates[0]
            if _edge_key(curr, nxt) in visited_edges:
                break
            prev, curr = curr, nxt

        return points

    # Trace from endpoints/junctions first
    for node in list(nodes):
        for neigh in neighbor_map.get(node, []):
            ek = _edge_key(node, neigh)
            if ek in visited_edges:
                continue
            poly = trace_from(node, neigh)
            if len(poly) >= 2:
                polylines.append(poly)

    # Trace remaining edges (loops)
    for p in coords:
        for neigh in neighbor_map.get(p, []):
            ek = _edge_key(p, neigh)
            if ek in visited_edges:
                continue

            start = p
            points = [start]
            prev = start
            curr = neigh
            while True:
                points.append(curr)
                visited_edges.add(_edge_key(prev, curr))
                neighs = neighbor_map.get(curr, [])
                next_candidates = [q for q in neighs if q != prev]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                if nxt == start:
                    break
                if _edge_key(curr, nxt) in visited_edges:
                    break
                prev, curr = curr, nxt

            if len(points) >= 2:
                polylines.append(points)

    return polylines


def rdp_simplify(points_xy, epsilon):
    if len(points_xy) < 3:
        return points_xy

    pts = np.asarray(points_xy, dtype=np.float32)
    keep = np.zeros((len(points_xy),), dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, len(points_xy) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        seg = pts[end] - pts[start]
        seg_len = float(np.hypot(seg[0], seg[1]))

        if seg_len == 0.0:
            d = pts[start + 1 : end] - pts[start]
            dists = np.hypot(d[:, 0], d[:, 1])
        else:
            v = pts[start + 1 : end] - pts[start]
            cross = np.abs(v[:, 0] * seg[1] - v[:, 1] * seg[0])
            dists = cross / seg_len

        idx = int(np.argmax(dists))
        max_dist = float(dists[idx])
        if max_dist > float(epsilon):
            real_idx = start + 1 + idx
            keep[real_idx] = True
            stack.append((start, real_idx))
            stack.append((real_idx, end))

    return [points_xy[i] for i in range(len(points_xy)) if keep[i]]


def largest_external_contour(binary_walls):
    contours, _ = cv2.findContours(binary_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def distance_to_contour_map(shape_hw, contour):
    if contour is None:
        return None
    h, w = shape_hw
    dist_input = np.full((h, w), 255, dtype=np.uint8)
    cv2.drawContours(dist_input, [contour], contourIdx=-1, color=0, thickness=1)
    return cv2.distanceTransform(dist_input, cv2.DIST_L2, 3)


def _fmt_num(n):
    if abs(n - round(n)) < 1e-6:
        return str(int(round(n)))
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _polyline_to_d(points_xy):
    if len(points_xy) < 2:
        return ""
    parts = [f"M{_fmt_num(points_xy[0][0])},{_fmt_num(points_xy[0][1])}"]
    for x, y in points_xy[1:]:
        parts.append(f"L{_fmt_num(x)},{_fmt_num(y)}")
    return " ".join(parts)


def write_stroke_svg(output_path, width, height, outer_d_parts, inner_d_parts, outer_width, inner_width):
    svg = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": str(int(width)),
        "height": str(int(height)),
        "viewBox": f"0 0 {int(width)} {int(height)}",
    })

    ET.SubElement(svg, f"{{{SVG_NS}}}rect", {
        "x": "0",
        "y": "0",
        "width": "100%",
        "height": "100%",
        "fill": "#ffffff",
    })

    common = {
        "fill": "none",
        "stroke": "#000000",
        "stroke-linecap": "round",
        "stroke-linejoin": "miter",
        "stroke-miterlimit": "4",
    }

    g_outer = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "outer", **common, "stroke-width": str(int(outer_width))})
    g_inner = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "interior", **common, "stroke-width": str(int(inner_width))})

    if outer_d_parts:
        ET.SubElement(g_outer, f"{{{SVG_NS}}}path", {"d": " ".join(outer_d_parts)})
    if inner_d_parts:
        ET.SubElement(g_inner, f"{{{SVG_NS}}}path", {"d": " ".join(inner_d_parts)})

    os.makedirs(os.path.dirname(os.path.abspath(str(output_path))), exist_ok=True)
    ET.ElementTree(svg).write(str(output_path), encoding="utf-8", xml_declaration=True)

# ---------------------------------------------------------------------------
# SVG path parsing and transform utilities
# ---------------------------------------------------------------------------
_PATH_CMD_RE = re.compile(
    r"([MmZzLlHhVvCcSsQqTtAa])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _tokenize_path(d):
    """Tokenize SVG path d attribute into list of (cmd, [float_args])."""
    tokens = []
    cmd = None
    args = []
    for match in _PATH_CMD_RE.finditer(d):
        c, n = match.groups()
        if c:
            if cmd is not None:
                tokens.append((cmd, args))
            cmd = c
            args = []
        elif n is not None:
            args.append(float(n))
    if cmd is not None:
        tokens.append((cmd, args))
    return tokens


def _fmt(n):
    """Format number for SVG path output."""
    if n == int(n) and abs(n) < 1e9:
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _rebuild_path(tokens):
    """Rebuild SVG path d attribute from tokens."""
    parts = []
    for cmd, args in tokens:
        if args:
            parts.append(cmd + " ".join(_fmt(a) for a in args))
        else:
            parts.append(cmd)
    return " ".join(parts)


def _transform_path(d, sx, sy, tx, ty):
    """
    Apply affine transform to SVG path coordinates.
    Absolute coords: (x*sx + tx, y*sy + ty)
    Relative coords: (dx*sx, dy*sy)
    """
    tokens = _tokenize_path(d)
    result = []

    for cmd, args in tokens:
        new_args = list(args)

        if cmd in ("Z", "z"):
            result.append((cmd, []))
            continue

        if cmd in ("M", "L", "T"):
            for i in range(0, len(args) - 1, 2):
                new_args[i] = args[i] * sx + tx
                new_args[i + 1] = args[i + 1] * sy + ty
        elif cmd in ("m", "l", "t"):
            for i in range(0, len(args) - 1, 2):
                new_args[i] = args[i] * sx
                new_args[i + 1] = args[i + 1] * sy
        elif cmd == "H":
            new_args = [a * sx + tx for a in args]
        elif cmd == "h":
            new_args = [a * sx for a in args]
        elif cmd == "V":
            new_args = [a * sy + ty for a in args]
        elif cmd == "v":
            new_args = [a * sy for a in args]
        elif cmd == "C":
            for i in range(0, len(args) - 5, 6):
                for j in (0, 2, 4):
                    new_args[i + j] = args[i + j] * sx + tx
                for j in (1, 3, 5):
                    new_args[i + j] = args[i + j] * sy + ty
        elif cmd == "c":
            for i in range(0, len(args) - 5, 6):
                for j in (0, 2, 4):
                    new_args[i + j] = args[i + j] * sx
                for j in (1, 3, 5):
                    new_args[i + j] = args[i + j] * sy
        elif cmd in ("S", "Q"):
            for i in range(0, len(args) - 3, 4):
                for j in (0, 2):
                    new_args[i + j] = args[i + j] * sx + tx
                for j in (1, 3):
                    new_args[i + j] = args[i + j] * sy + ty
        elif cmd in ("s", "q"):
            for i in range(0, len(args) - 3, 4):
                for j in (0, 2):
                    new_args[i + j] = args[i + j] * sx
                for j in (1, 3):
                    new_args[i + j] = args[i + j] * sy
        elif cmd == "A":
            for i in range(0, len(args) - 6, 7):
                new_args[i] = args[i] * abs(sx)
                new_args[i + 1] = args[i + 1] * abs(sy)
                if sy < 0:
                    new_args[i + 4] = 1.0 - args[i + 4]
                new_args[i + 5] = args[i + 5] * sx + tx
                new_args[i + 6] = args[i + 6] * sy + ty
        elif cmd == "a":
            for i in range(0, len(args) - 6, 7):
                new_args[i] = args[i] * abs(sx)
                new_args[i + 1] = args[i + 1] * abs(sy)
                if sy < 0:
                    new_args[i + 4] = 1.0 - args[i + 4]
                new_args[i + 5] = args[i + 5] * sx
                new_args[i + 6] = args[i + 6] * sy

        result.append((cmd, new_args))

    return _rebuild_path(result)


def _path_bounding_box(d):
    """Compute approximate bounding box by tracking pen position."""
    tokens = _tokenize_path(d)
    xs, ys = [], []
    cx, cy = 0.0, 0.0
    start_x, start_y = 0.0, 0.0

    for cmd, args in tokens:
        if cmd == "M":
            for i in range(0, len(args) - 1, 2):
                cx, cy = args[i], args[i + 1]
                xs.append(cx)
                ys.append(cy)
            start_x, start_y = cx, cy
        elif cmd == "m":
            for i in range(0, len(args) - 1, 2):
                cx += args[i]
                cy += args[i + 1]
                xs.append(cx)
                ys.append(cy)
            start_x, start_y = cx, cy
        elif cmd == "L":
            for i in range(0, len(args) - 1, 2):
                cx, cy = args[i], args[i + 1]
                xs.append(cx)
                ys.append(cy)
        elif cmd == "l":
            for i in range(0, len(args) - 1, 2):
                cx += args[i]
                cy += args[i + 1]
                xs.append(cx)
                ys.append(cy)
        elif cmd == "H":
            for v in args:
                cx = v
                xs.append(cx)
        elif cmd == "h":
            for v in args:
                cx += v
                xs.append(cx)
        elif cmd == "V":
            for v in args:
                cy = v
                ys.append(cy)
        elif cmd == "v":
            for v in args:
                cy += v
                ys.append(cy)
        elif cmd == "C":
            for i in range(0, len(args) - 5, 6):
                xs.extend([args[i], args[i + 2], args[i + 4]])
                ys.extend([args[i + 1], args[i + 3], args[i + 5]])
                cx, cy = args[i + 4], args[i + 5]
        elif cmd == "c":
            for i in range(0, len(args) - 5, 6):
                xs.extend([cx + args[i], cx + args[i + 2], cx + args[i + 4]])
                ys.extend([cy + args[i + 1], cy + args[i + 3], cy + args[i + 5]])
                cx += args[i + 4]
                cy += args[i + 5]
        elif cmd in ("S", "Q"):
            for i in range(0, len(args) - 3, 4):
                xs.extend([args[i], args[i + 2]])
                ys.extend([args[i + 1], args[i + 3]])
                cx, cy = args[i + 2], args[i + 3]
        elif cmd in ("s", "q"):
            for i in range(0, len(args) - 3, 4):
                xs.extend([cx + args[i], cx + args[i + 2]])
                ys.extend([cy + args[i + 1], cy + args[i + 3]])
                cx += args[i + 2]
                cy += args[i + 3]
        elif cmd == "T":
            for i in range(0, len(args) - 1, 2):
                cx, cy = args[i], args[i + 1]
                xs.append(cx)
                ys.append(cy)
        elif cmd == "t":
            for i in range(0, len(args) - 1, 2):
                cx += args[i]
                cy += args[i + 1]
                xs.append(cx)
                ys.append(cy)
        elif cmd == "A":
            for i in range(0, len(args) - 6, 7):
                cx, cy = args[i + 5], args[i + 6]
                xs.append(cx)
                ys.append(cy)
        elif cmd == "a":
            for i in range(0, len(args) - 6, 7):
                cx += args[i + 5]
                cy += args[i + 6]
                xs.append(cx)
                ys.append(cy)
        elif cmd in ("Z", "z"):
            cx, cy = start_x, start_y

    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------

def detect_input_type(filepath):
    """Detect input file type. Returns 'raster', 'svg', 'pdf', or None."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in SUPPORTED_RASTER:
        return "raster"
    elif ext in SUPPORTED_VECTOR:
        return "svg"
    elif ext in SUPPORTED_PDF:
        return "pdf"
    return None


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def pdf_to_svg(pdf_path, output_svg_path):
    """Convert PDF to SVG using pdf2svg. Returns True on success."""
    cmd = [PDF2SVG_BIN, pdf_path, output_svg_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error(f"  pdf2svg failed: {result.stderr}")
            return False
        log.info(f"  Converted PDF to SVG via pdf2svg")
        return True
    except FileNotFoundError:
        log.error(
            "  Error: 'pdf2svg' not found.\n"
            "  Install: Windows: download from GitHub\n"
            "           Mac: brew install pdf2svg\n"
            "           Linux: sudo apt install pdf2svg"
        )
        return False
    except subprocess.TimeoutExpired:
        log.error("  pdf2svg timed out (>120s)")
        return False


# ---------------------------------------------------------------------------
# Raster preprocessing
# ---------------------------------------------------------------------------

def convert_to_grayscale(input_path, debug_dir=None):
    """Step 1: Load image and convert to grayscale."""
    img = cv2.imread(input_path)
    if img is None:
        log.error(f"Error: Could not load image '{input_path}'")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if debug_dir:
        p = os.path.join(debug_dir, "_01_gray.png")
        cv2.imwrite(p, gray)
        log.info(f"  [debug] Saved: {p}")

    return gray


def apply_threshold(gray, threshold_value, debug_dir=None):
    """Step 2: Adaptive binary threshold with morphological closing."""
    # Adaptive Gaussian threshold — walls BLACK (0), background WHITE (255)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=71,
        C=6
    )

    if debug_dir:
        p = os.path.join(debug_dir, "_02_threshold.png")
        cv2.imwrite(p, binary)
        log.info(f"  [debug] Saved: {p}")

    # Morphological closing to reconnect broken wall lines
    # Invert so walls=255 (foreground) for closing, then invert back
    kernel = np.ones((3, 3), np.uint8)
    inverted = cv2.bitwise_not(binary)
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.bitwise_not(closed)

    return binary


def remove_small_components(binary, min_size, debug_dir=None):
    """Step 3: Remove small components + dilate walls."""
    # Input: walls BLACK (0), background WHITE (255)
    # Invert so walls=255 for component analysis
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)
    removed = 0
    kept = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == label_id] = 255
            kept += 1
        else:
            removed += 1

    log.info(f"  Components: {kept} kept, {removed} removed (area < {min_size})")

    # Dilate to solidify walls before Potrace
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Invert back: walls BLACK (0), background WHITE (255)
    result = cv2.bitwise_not(cleaned)

    if debug_dir:
        p = os.path.join(debug_dir, "_03_cleaned.png")
        cv2.imwrite(p, result)
        log.info(f"  [debug] Saved: {p}")

    return result


def morphological_close(binary, debug_dir=None):
    """Step 4: Morphological closing (3x3) on walls to fill small gaps."""
    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.bitwise_not(binary)
    walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed = cv2.bitwise_not(walls_closed)

    if debug_dir:
        p = os.path.join(debug_dir, "step4_closed.png")
        cv2.imwrite(p, closed)
        log.info(f"  [debug] Saved: {p}")

    return closed


# ---------------------------------------------------------------------------
# Wall standardization
# ---------------------------------------------------------------------------

def _morph_skeleton(binary_fg):
    """
    Morphological skeletonization (Zhang-Suen style via OpenCV ops).
    Input: binary with foreground=255, background=0.
    Returns: skeleton (foreground=255, background=0).
    """
    skel = np.zeros_like(binary_fg)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_fg.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _find_exterior_mask(binary):
    """
    Find the exterior region by flood-filling from image borders.
    Input: binary with walls=black(0), background=white(255).
    Uses aggressive closing on the wall mask first to prevent leaking through door gaps.
    Returns: exterior mask (exterior=255, else=0).
    """
    h, w = binary.shape

    # Aggressive close to seal door openings for flood fill only.
    # Important: close the wall mask (foreground) rather than the background,
    # otherwise the operation can erase walls and cause flood-fill to leak.
    kernel = np.ones((7, 7), np.uint8)
    walls = cv2.bitwise_not(binary)
    walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel, iterations=3)
    closed = cv2.bitwise_not(walls_closed)

    flood = closed.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    fill_val = 128

    for x in range(w):
        if flood[0, x] == 255:
            cv2.floodFill(flood, mask, (x, 0), fill_val)
        if flood[h - 1, x] == 255:
            cv2.floodFill(flood, mask, (x, h - 1), fill_val)
    for y in range(h):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, mask, (0, y), fill_val)
        if flood[y, w - 1] == 255:
            cv2.floodFill(flood, mask, (w - 1, y), fill_val)

    exterior = (flood == fill_val).astype(np.uint8) * 255
    return exterior


def skeletonize_and_standardize(binary, outer_thickness, inner_thickness, debug_dir=None):
    """
    Steps 5-7: Skeletonize walls, detect outer vs interior, dilate to
    standardized thicknesses.

    Input: binary with walls=black(0), background=white(255).
    Returns: standardized binary (walls=black, background=white).
    """
    # Invert: walls = 255 (foreground for skeletonization)
    wall_fg = cv2.bitwise_not(binary)

    # Step 5: Skeletonize
    log.info("  Skeletonizing walls to centerlines...")
    skeleton = _morph_skeleton(wall_fg)

    if debug_dir:
        p = os.path.join(debug_dir, "step5_skeleton.png")
        cv2.imwrite(p, skeleton)
        log.info(f"  [debug] Saved: {p}")

    # Step 6: Detect outer vs interior
    log.info("  Detecting outer boundary vs interior walls...")
    exterior = _find_exterior_mask(binary)

    if debug_dir:
        p = os.path.join(debug_dir, "step6_exterior.png")
        cv2.imwrite(p, exterior)
        log.info(f"  [debug] Saved: {p}")

    # Determine dilation radius to reach skeleton from exterior
    dist = cv2.distanceTransform(wall_fg, cv2.DIST_L2, 5)
    max_half = int(np.max(dist)) + 3 if np.max(dist) > 0 else 10
    max_half = max(max_half, 10)

    # Dilate exterior to overlap with outer wall skeleton
    k = 2 * max_half + 1
    big_kernel = np.ones((k, k), np.uint8)
    exterior_dilated = cv2.dilate(exterior, big_kernel, iterations=1)

    # Check if there's a meaningful outer boundary
    non_wall = np.sum(binary == 255)
    ext_pixels = np.sum(exterior == 255)

    if non_wall > 0 and ext_pixels / non_wall > 0.9:
        # No enclosed building — treat all walls as interior
        log.info("  No clear outer boundary detected; using uniform thickness")
        outer_skel = np.zeros_like(skeleton)
        inner_skel = skeleton.copy()
    else:
        outer_skel = cv2.bitwise_and(skeleton, exterior_dilated)
        inner_skel = cv2.bitwise_and(skeleton, cv2.bitwise_not(outer_skel))

    outer_count = cv2.countNonZero(outer_skel)
    inner_count = cv2.countNonZero(inner_skel)
    log.info(f"  Outer boundary: {outer_count} skeleton px, Interior: {inner_count} skeleton px")

    if debug_dir:
        # Color-coded debug image: outer=red, interior=blue
        vis = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        vis[outer_skel == 255] = (0, 0, 255)   # red in BGR
        vis[inner_skel == 255] = (255, 0, 0)    # blue in BGR
        p = os.path.join(debug_dir, "step6_wall_classes.png")
        cv2.imwrite(p, vis)
        log.info(f"  [debug] Saved: {p}")

    # Step 7: Dilate each to standardized thickness
    log.info(f"  Standardizing: outer={outer_thickness}px, interior={inner_thickness}px")

    standardized = np.zeros_like(wall_fg)

    if outer_count > 0:
        r = max(outer_thickness // 2, 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        outer_dilated = cv2.dilate(outer_skel, kern, iterations=1)
        standardized = cv2.bitwise_or(standardized, outer_dilated)

    if inner_count > 0:
        r = max(inner_thickness // 2, 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner_dilated = cv2.dilate(inner_skel, kern, iterations=1)
        standardized = cv2.bitwise_or(standardized, inner_dilated)

    # Invert back: walls = black, background = white
    result = cv2.bitwise_not(standardized)

    if debug_dir:
        p = os.path.join(debug_dir, "step7_standardized.png")
        cv2.imwrite(p, result)
        log.info(f"  [debug] Saved: {p}")

    return result


# ---------------------------------------------------------------------------
# Potrace tracing
# ---------------------------------------------------------------------------

def trace_to_svg(binary_image, debug_dir=None):
    """
    Step 8: Trace binary image to SVG using Potrace (fill mode).
    alphamax=0, opttolerance=0.
    Returns path to generated SVG file.
    """
    if debug_dir:
        work_dir = debug_dir
    else:
        # Default temp dirs may be locked down in some environments.
        # Prefer a scratch folder inside the project output folder.
        work_dir = _make_work_dir(SCRATCH_ROOT, "potrace_")

    bmp_path = os.path.join(work_dir, "trace_input.bmp")
    svg_path = os.path.join(work_dir, "floorplan_raw.svg")

    # Invert: walls (black) become foreground (white) for Potrace
    inverted = cv2.bitwise_not(binary_image)
    cv2.imwrite(bmp_path, inverted)

    cmd = [
        POTRACE_BIN,
        bmp_path,
        "-s",             # SVG output
        "-o", svg_path,
        "-t", "5",        # turdsize: suppress speckles up to 5px
        "-a", "0",        # alphamax=0: sharp corners
        "-O", "0",        # opttolerance=0: no curve optimization
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error(f"Error: potrace failed:\n{result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        log.error(
            "Error: 'potrace' not found.\n"
            "Install: Windows: download from https://potrace.sourceforge.net\n"
            "         Mac: brew install potrace\n"
            "         Linux: sudo apt install potrace"
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        log.error("Error: potrace timed out (>120s)")
        sys.exit(1)

    if not os.path.exists(svg_path):
        log.error("Error: potrace did not produce output SVG.")
        sys.exit(1)

    if debug_dir:
        log.info(f"  [debug] Saved raw SVG: {svg_path}")

    if not debug_dir:
        os.remove(bmp_path)

    return svg_path


# ---------------------------------------------------------------------------
# SVG processing
# ---------------------------------------------------------------------------

def _parse_potrace_transform(transform_str):
    """
    Parse Potrace's transform attribute.
    Returns (scale_x, scale_y, translate_x, translate_y).
    """
    sx, sy, tx, ty = 1.0, 1.0, 0.0, 0.0

    # Extract translate
    t_match = re.search(r"translate\(([^,)]+)[,\s]+([^)]+)\)", transform_str)
    if t_match:
        tx = float(t_match.group(1))
        ty = float(t_match.group(2))

    # Extract scale
    s_match = re.search(r"scale\(([^,)]+)[,\s]+([^)]+)\)", transform_str)
    if s_match:
        sx = float(s_match.group(1))
        sy = float(s_match.group(2))

    return sx, sy, tx, ty


def remove_transforms_and_style(tree, debug_dir=None, outer_width=8, inner_width=4):
    """
    Normalize an SVG to RoomView stroke-only style:
    - Apply and remove a simple <g transform="..."> wrapper (scale/translate)
    - Set stroke="#000000", fill="none" on all paths
    - Remove text elements
    - Set white background
    - Convert width/height from pt to px
    - Remove all opacity/dashed/vector-effect attributes
    - Standardize thickness: outer=outer_width, interior=inner_width (heuristic)
    """
    root = tree.getroot()

    # --- Convert width/height from pt to px ---
    for attr in ("width", "height"):
        val = root.get(attr, "")
        num = re.sub(r"[a-zA-Z]+$", "", val.strip())
        try:
            root.set(attr, f"{float(num):.0f}")
        except ValueError:
            pass

    # --- Find and apply the <g transform="..."> from Potrace ---
    ns_g = f"{{{SVG_NS}}}g"
    ns_path = f"{{{SVG_NS}}}path"

    g_elements = [e for e in root if e.tag == ns_g or e.tag == "g"]

    for g in g_elements:
        transform = g.get("transform", "")
        if not transform:
            continue

        sx, sy, tx, ty = _parse_potrace_transform(transform)
        log.info(f"  Removing transform: scale=({sx},{sy}) translate=({tx},{ty})")

        # Transform all path coordinates inside this <g>
        for path_elem in g.iter():
            if path_elem.tag == ns_path or path_elem.tag == "path":
                d = path_elem.get("d", "")
                if d:
                    path_elem.set("d", _transform_path(d, sx, sy, tx, ty))

        # Remove transform attribute
        del g.attrib["transform"]

    # --- Remove text elements ---
    parent_map = {}
    for parent in root.iter():
        for child in parent:
            parent_map[child] = parent

    removals = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in ("text", "tspan", "textPath"):
            if elem in parent_map:
                removals.append((parent_map[elem], elem))

    for parent, child in removals:
        try:
            parent.remove(child)
        except ValueError:
            pass

    # --- Set white background ---
    root.set("style", "background-color: white;")
    bg = ET.Element(f"{{{SVG_NS}}}rect", {
        "x": "0",
        "y": "0",
        "width": "100%",
        "height": "100%",
        "fill": "#ffffff",
    })
    root.insert(0, bg)

    # --- Clean group elements ---
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "g":
            for attr in ("fill", "stroke", "stroke-opacity", "opacity", "style"):
                if attr in elem.attrib:
                    del elem.attrib[attr]

    # --- Style all paths: stroke-only ---
    path_count = 0
    for elem in root.iter():
        if elem.tag == ns_path or elem.tag == "path":
            elem.set("fill", "none")
            elem.set("stroke", "#000000")
            elem.set("stroke-linecap", "round")
            elem.set("stroke-linejoin", "miter")
            elem.set("stroke-miterlimit", "4")
            for attr in ("style", "opacity", "vector-effect", "stroke-width",
                         "stroke-opacity", "stroke-dasharray", "stroke-dashoffset",
                         "fill-opacity", "fill-rule"):
                if attr in elem.attrib:
                    del elem.attrib[attr]
            path_count += 1

    log.info(f"  Styled {path_count} paths (stroke=#000000, fill=none)")

    # --- Detect outer boundary for logging ---
    all_paths = [e for e in root.iter() if e.tag == ns_path or e.tag == "path"]
    outer_idx = _detect_and_log_outer(all_paths, root)

    # --- Standardize stroke widths ---
    for idx, elem in enumerate(all_paths):
        w = outer_width if (outer_idx is not None and idx == outer_idx) else inner_width
        elem.set("stroke-width", str(int(w)))

    if debug_dir:
        p = os.path.join(debug_dir, "step9_styled.svg")
        tree.write(p, xml_declaration=True, encoding="unicode")
        log.info(f"  [debug] Saved: {p}")

    return tree


def _detect_and_log_outer(path_elements, root):
    """Detect outer boundary path and log results. Returns best index, or None."""
    if not path_elements:
        return None

    # Get SVG dimensions
    w_str = re.sub(r"[a-zA-Z]+$", "", root.get("width", "0").strip())
    h_str = re.sub(r"[a-zA-Z]+$", "", root.get("height", "0").strip())
    try:
        svg_w, svg_h = float(w_str), float(h_str)
    except ValueError:
        svg_w, svg_h = 0.0, 0.0

    margin = 10.0
    best_idx = -1
    best_area = 0

    for idx, elem in enumerate(path_elements):
        d = elem.get("d", "")
        if not d:
            continue
        bbox = _path_bounding_box(d)
        if bbox is None:
            continue
        min_x, min_y, max_x, max_y = bbox
        area = (max_x - min_x) * (max_y - min_y)

        touches = (
            min_x <= margin
            or min_y <= margin
            or (svg_w > 0 and max_x >= svg_w - margin)
            or (svg_h > 0 and max_y >= svg_h - margin)
        )

        if touches and area > best_area:
            best_area = area
            best_idx = idx

    if best_idx >= 0:
        log.info(f"  Outer boundary: path {best_idx + 1} of {len(path_elements)} (area={best_area:.0f})")
    else:
        log.info(f"  Outer boundary: not detected ({len(path_elements)} paths total)")
        return None

    return best_idx


def save_svg(tree, output_path):
    """Save final SVG."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    tree.write(output_path, xml_declaration=True, encoding="unicode")
    log.info(f"  Output saved: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _find_input_files(input_dir):
    """Find all supported files in a directory."""
    files = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_RASTER + SUPPORTED_VECTOR + SUPPORTED_PDF:
            files.append(os.path.join(input_dir, fname))
    return files


def _process_raster_bgr(bgr, output_path, args):
    """Process BGR image directly (used by PDF and embedded SVG rasters)."""
    import tempfile
    # Save BGR to temp file and process through normal pipeline
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, bgr)
    
    try:
        process_raster(tmp_path, output_path, args, debug_dir=None)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def _process_raster_bgr(bgr, output_path, args):
    """
    Raster pipeline:
      grayscale -> contrast boost -> adaptive threshold -> close -> area filter ->
      thinning -> skeleton polylines -> outer/interior classify -> stroke SVG.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _write_png(_debug_path(output_path, "_01_gray.png"), gray)

    gray = boost_contrast(gray)
    binary = adaptive_binarize(gray, args.thresh_blocksize, args.thresh_C)
    _write_png(_debug_path(output_path, "_02_binary.png"), binary)

    binary = morph_close(binary, args.close_kernel)
    cleaned = remove_small_components_fg(binary, args.min_component_area)
    _write_png(_debug_path(output_path, "_03_cleaned.png"), cleaned)

    log.info("  Skeletonizing (Zhang-Suen thinning)...")
    skeleton = zhang_suen_thinning(cleaned)
    _write_png(_debug_path(output_path, "_04_skeleton.png"), skeleton)

    contour = largest_external_contour(cleaned)
    dist_map = distance_to_contour_map(cleaned.shape, contour)
    if dist_map is None:
        log.info("  Outer contour: not detected (treating all as interior)")

    log.info("  Vectorizing skeleton -> polylines...")
    polylines = skeleton_to_polylines(skeleton)
    log.info(f"  Polylines: {len(polylines)}")

    outer_d_parts = []
    inner_d_parts = []
    epsilon = 1.0  # RDP simplification (pixels)

    for poly in polylines:
        if len(poly) < 2:
            continue

        is_outer = False
        if dist_map is not None:
            min_d = float("inf")
            for y, x in poly:
                if 0 <= y < dist_map.shape[0] and 0 <= x < dist_map.shape[1]:
                    d = float(dist_map[y, x])
                    if d < min_d:
                        min_d = d
                        if min_d <= float(args.outer_band):
                            break
            is_outer = min_d <= float(args.outer_band)

        pts_xy = [(float(x) + 0.5, float(y) + 0.5) for (y, x) in poly]
        pts_xy = rdp_simplify(pts_xy, epsilon=epsilon)
        d = _polyline_to_d(pts_xy)
        if not d:
            continue

        if is_outer:
            outer_d_parts.append(d)
        else:
            inner_d_parts.append(d)

    h, w = cleaned.shape
    write_stroke_svg(
        output_path,
        width=w,
        height=h,
        outer_d_parts=outer_d_parts,
        inner_d_parts=inner_d_parts,
        outer_width=args.outer_width,
        inner_width=args.inner_width,
    )

    final_copy = _debug_path(output_path, "_final.svg")
    if os.path.abspath(str(output_path)).lower() != os.path.abspath(str(final_copy)).lower():
        write_stroke_svg(
            final_copy,
            width=w,
            height=h,
            outer_d_parts=outer_d_parts,
            inner_d_parts=inner_d_parts,
            outer_width=args.outer_width,
            inner_width=args.inner_width,
        )


def process_raster(input_path, output_path, args, debug_dir=None):
    """Raster pipeline: adaptive threshold -> clean -> Potrace -> style SVG."""
    log.info("[1/4] Converting to grayscale...")
    gray = convert_to_grayscale(input_path, debug_dir=debug_dir)

    log.info("[2/4] Adaptive threshold + morphological close...")
    binary = apply_threshold(gray, args.threshold, debug_dir=debug_dir)

    log.info(f"[3/4] Removing noise (< 250px)...")
    binary = remove_small_components(binary, 250, debug_dir=debug_dir)

    log.info("[4/4] Tracing to vector (Potrace)...")
    raw_svg = trace_to_svg(binary, debug_dir=debug_dir)

    log.info("Styling SVG...")
    tree = ET.parse(raw_svg)
    tree = remove_transforms_and_style(
        tree,
        debug_dir=debug_dir,
        outer_width=args.outer_width,
        inner_width=args.inner_width,
    )

    save_svg(tree, output_path)

    log.info("  ✓ Complete")

    # Clean up temp files
    if not debug_dir and os.path.exists(raw_svg):
        temp_dir = os.path.dirname(raw_svg)
        try:
            os.remove(raw_svg)
            os.rmdir(temp_dir)
        except OSError:
            pass


def process_svg(input_path, output_path, args, debug_dir=None):
    """SVG pipeline: normalize vector, or treat embedded raster as raster."""
    tree = ET.parse(input_path)
    if svg_contains_embedded_raster(tree):
        log.info("  SVG contains embedded raster; treating as raster.")
        img = extract_embedded_svg_image(input_path, tree)
        if img is None:
            raise RuntimeError("SVG contains <image> but embedded raster could not be extracted.")
        _process_raster_bgr(img, output_path, args)
        return

    log.info("[1/1] Normalizing SVG styles (stroke-only)...")
    tree = remove_transforms_and_style(
        tree,
        debug_dir=debug_dir,
        outer_width=args.outer_width,
        inner_width=args.inner_width,
    )
    save_svg(tree, output_path)

    final_copy = _debug_path(output_path, "_final.svg")
    if os.path.abspath(str(output_path)).lower() != os.path.abspath(str(final_copy)).lower():
        save_svg(tree, final_copy)


def process_pdf(input_path, output_path, args, debug_dir=None):
    """PDF pipeline: render to high-res raster -> raster pipeline."""
    log.info(f"  Rendering PDF at {args.dpi} DPI...")
    bgr = render_pdf_first_page(input_path, dpi=args.dpi)
    _process_raster_bgr(bgr, output_path, args)
    return True

    log.info("[1/2] Converting PDF to SVG...")

    if debug_dir:
        work_dir = debug_dir
    else:
        work_dir = _make_work_dir(SCRATCH_ROOT, "pdf_")
    temp_svg = os.path.join(work_dir, "pdf_converted.svg")

    if not pdf_to_svg(input_path, temp_svg):
        log.error("  Failed to convert PDF. Skipping.")
        return False

    log.info("[2/2] Normalizing SVG styles...")

    tree = ET.parse(temp_svg)
    tree = remove_transforms_and_style(
        tree,
        debug_dir=debug_dir,
        outer_width=args.outer_width,
        inner_width=args.inner_width,
    )

    save_svg(tree, output_path)

    log.info("  Standardization: styles only (no Potrace)")
    log.info("  Potrace: not used")

    if not debug_dir:
        try:
            os.remove(temp_svg)
            os.rmdir(work_dir)
        except OSError:
            pass


def run_pipeline(input_path, output_path, args, debug_dir=None):
    """Detect input type and run the appropriate pipeline. Returns True on success."""
    input_type = detect_input_type(input_path)

    if input_type is None:
        ext = os.path.splitext(input_path)[1]
        log.error(f"Error: Unsupported file type '{ext}'")
        return False

    log.info(f"  Input type: {input_type.upper()}")

    if input_type == "raster":
        process_raster(input_path, output_path, args, debug_dir=debug_dir)
    elif input_type == "svg":
        process_svg(input_path, output_path, args, debug_dir=debug_dir)
    elif input_type == "pdf":
        result = process_pdf(input_path, output_path, args, debug_dir=debug_dir)
        if result is False:
            return False

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    input_dir_default = os.path.join(SCRIPT_DIR, "input")
    output_dir_default = os.path.join(SCRIPT_DIR, "output")

    parser = argparse.ArgumentParser(
        description="RoomView Floor Plan Normalization — "
                    "Standardized drafting-style SVG from any floor plan source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python normalize_floorplan.py                                (batch: all files in input/)
  python normalize_floorplan.py floor3.png                     (single raster file)
  python normalize_floorplan.py plan.pdf                       (PDF input)
  python normalize_floorplan.py drawing.svg                    (SVG input)
  python normalize_floorplan.py floor.png --threshold 180 --debug
  python normalize_floorplan.py floor.png --outer-thickness 8 --inner-thickness 4
        """,
    )

    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to input file (PNG/JPG/SVG/PDF). If omitted, batch processes input/.",
    )
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output SVG path (single-file) or output directory (batch). Default: output/.",
    )
    parser.add_argument(
        "--dpi", type=int, default=450,
        help="PDF rasterization DPI (300-600 typical).",
    )
    parser.add_argument(
        "--thresh_blocksize", type=int, default=51,
        help="Adaptive threshold block size (odd).",
    )
    parser.add_argument(
        "--thresh_C", type=int, default=5,
        help="Adaptive threshold C (lower -> more black).",
    )
    parser.add_argument(
        "--close_kernel", type=int, default=3,
        help="Morphological close kernel size (pixels).",
    )
    parser.add_argument(
        "--min_component_area", type=int, default=80,
        help="Minimum connected component area to keep (removes icons/text).",
    )
    parser.add_argument(
        "--outer_width", type=int, default=8,
        help="Outer wall stroke width (px).",
    )
    parser.add_argument(
        "--inner_width", type=int, default=4,
        help="Interior wall stroke width (px).",
    )
    parser.add_argument(
        "--outer_band", type=int, default=12,
        help="Pixels from outer contour to classify as outer wall.",
    )

    args = parser.parse_args()

    if args.input:
        # --- Single file mode ---
        if not os.path.exists(args.input):
            log.error(f"Error: Input file '{args.input}' not found.")
            sys.exit(1)

        if detect_input_type(args.input) is None:
            log.error(f"Error: Unsupported file type.")
            sys.exit(1)

        base = os.path.splitext(os.path.basename(args.input))[0]
        if args.output:
            output_path = args.output
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"{base}.svg")
        else:
            output_path = os.path.join(output_dir_default, f"{base}.svg")

        log.info("RoomView Floor Plan Normalization")
        log.info("=" * 50)
        run_pipeline(args.input, output_path, args, debug_dir=None)

        log.info("\nDone.")

    else:
        # --- Batch mode ---
        if not os.path.isdir(input_dir_default):
            os.makedirs(input_dir_default, exist_ok=True)

        files = _find_input_files(input_dir_default)

        if not files:
            log.info(f"No files found in: {input_dir_default}")
            log.info("Drop your floor plan files (PNG/JPG/SVG/PDF) into input/ and run again.")
            sys.exit(0)

        out_dir = args.output if args.output else output_dir_default
        if isinstance(out_dir, str) and out_dir.lower().endswith(".svg"):
            out_dir = os.path.dirname(out_dir) or output_dir_default
        os.makedirs(out_dir, exist_ok=True)

        log.info("RoomView Floor Plan Normalization — Batch Mode")
        log.info("=" * 60)
        log.info(f"  Input folder:  {input_dir_default}")
        log.info(f"  Output folder: {out_dir}")
        log.info(f"  Files found:   {len(files)}")

        succeeded = 0
        failed = []

        for i, filepath in enumerate(files, 1):
            base = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(out_dir, f"{base}.svg")

            log.info(f"\n{'—' * 50}")
            log.info(f"  [{i}/{len(files)}] {os.path.basename(filepath)}")
            log.info(f"{'—' * 50}")

            try:
                ok = run_pipeline(filepath, output_path, args, debug_dir=None)
                if ok:
                    succeeded += 1
                else:
                    failed.append(os.path.basename(filepath))
            except Exception as exc:
                log.error(f"  Error processing file: {exc}")
                failed.append(os.path.basename(filepath))

        log.info(f"\n{'=' * 50}")
        log.info(f"Batch complete. {succeeded}/{len(files)} files processed successfully.")
        if failed:
            log.info(f"  Skipped ({len(failed)}): {', '.join(failed)}")
        log.info(f"Output: {out_dir}")


def _create_overlay(input_image_path, svg_path):
    """Generate overlay PNG for visual QC."""
    from PIL import Image

    original = Image.open(input_image_path).convert("RGBA")
    width, height = original.size

    overlay_base = original.copy()
    alpha = overlay_base.split()[3]
    alpha = alpha.point(lambda p: int(p * 0.5))
    overlay_base.putalpha(alpha)

    background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    background = Image.alpha_composite(background, overlay_base)

    overlay_path = os.path.splitext(svg_path)[0] + "_overlay.png"

    try:
        import cairosvg
        svg_uri = Path(os.path.abspath(svg_path)).as_uri()
        svg_png = cairosvg.svg2png(url=svg_uri, output_width=width, output_height=height)
        from io import BytesIO
        svg_layer = Image.open(BytesIO(svg_png)).convert("RGBA")
        result = Image.alpha_composite(background, svg_layer)
        result.save(overlay_path)
        log.info(f"  Overlay saved: {overlay_path}")
    except ImportError:
        background.save(overlay_path)
        log.info(f"  Overlay saved (without SVG render): {overlay_path}")
        log.info("  Install cairosvg for full overlay: pip install cairosvg")


if __name__ == "__main__":
    main()
