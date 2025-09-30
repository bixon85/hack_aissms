import io
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.feature import blob_log
from scipy.signal import savgol_filter


def _read_image_bytes(image_bytes: bytes) -> np.ndarray:
	img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _detect_dots(image_bgr: np.ndarray, *, min_area: int = 12, max_area: int = 6000, hough_param2: int = 18) -> Tuple[List[Tuple[int, int]], float]:
	# Contrast Limited AHE for robust lighting
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	l = clahe.apply(l)
	img_clahe = cv2.merge((l, a, b))
	image_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
	kernel = np.ones((3, 3), np.uint8)
	th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

	params = cv2.SimpleBlobDetector_Params()
	params.filterByArea = True
	params.minArea = float(min_area)
	params.maxArea = float(max_area)
	params.filterByCircularity = False
	params.filterByInertia = False
	params.filterByConvexity = False
	params.filterByColor = True
	params.blobColor = 255
	detector = cv2.SimpleBlobDetector_create(params)
	keypoints = detector.detect(th)
	dots = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]

	# Hough fallback
	circles = cv2.HoughCircles(
		gray,
		cv2.HOUGH_GRADIENT,
		dp=1.2,
		minDist=10,
		param1=120,
		param2=int(hough_param2),
		minRadius=3,
		maxRadius=20,
	)
	if circles is not None:
		for x, y, r in np.uint16(np.around(circles))[0, :]:
			dots.append((int(x), int(y)))

	# Merge close detections
	if dots:
		pts = np.array(dots)
		keep = []
		used = set()
		for i in range(len(pts)):
			if i in used:
				continue
			close = np.where(np.hypot(pts[:, 0]-pts[i, 0], pts[:, 1]-pts[i, 1]) < 6)[0]
			used.update(close.tolist())
			cx = int(np.mean(pts[close, 0]))
			cy = int(np.mean(pts[close, 1]))
			keep.append((cx, cy))
		dots = keep

	# spacing estimate via NN
	avg_spacing = 0.0
	if len(dots) > 1:
		pts = np.array(dots)
		dmin_list = []
		for i in range(len(pts)):
			delta = pts - pts[i]
			delta = delta[np.any(delta != 0, axis=1)]
			if len(delta) == 0:
				continue
			dmin_list.append(np.min(np.hypot(delta[:, 0], delta[:, 1])))
		avg_spacing = float(np.median(dmin_list)) if dmin_list else 0.0

	return dots, avg_spacing


def _snap_to_grid(dots: List[Tuple[int, int]], tol_frac: float = 0.35) -> List[Tuple[int, int]]:
	# Try to regularize to nearest row/col band using median spacing
	if len(dots) < 4:
		return dots
	xs = np.array([x for x, _ in dots])
	ys = np.array([y for _, y in dots])
	# band centers by 1D clustering via rounding to spacing
	# estimate spacing as robust median of diffs
	dx = np.diff(np.sort(xs))
	dy = np.diff(np.sort(ys))
	spacing = np.median(np.concatenate([dx[dx > 0], dy[dy > 0]])) if len(dx) + len(dy) > 0 else 0
	if spacing <= 0:
		return dots
	tol = max(3, int(spacing * tol_frac))
	def snap(vals):
		vals_sorted = np.sort(vals)
		bands = [vals_sorted[0]]
		for v in vals_sorted[1:]:
			if abs(v - bands[-1]) > tol:
				bands.append(v)
		return np.array(bands)
	xbands = snap(xs)
	ybands = snap(ys)
	# snap each dot to nearest band center
	snapped = []
	for (x, y) in dots:
		xc = int(xbands[np.argmin(np.abs(xbands - x))])
		yc = int(ybands[np.argmin(np.abs(ybands - y))])
		snapped.append((xc, yc))
	return snapped


def _estimate_grid(dots: List[Tuple[int, int]]) -> Tuple[str, Tuple[int, int]]:
	if not dots:
		return "0x0", (0, 0)
	xs = sorted(d[0] for d in dots)
	ys = sorted(d[1] for d in dots)
	def count_bands(vals: List[int], tol: int = 12) -> int:
		bands = 1
		for i in range(1, len(vals)):
			if abs(vals[i] - vals[i - 1]) > tol:
				bands += 1
		return bands
	cols = count_bands(xs)
	rows = count_bands(ys)
	return f"{rows}x{cols}", (rows, cols)


def _estimate_symmetry(image_bgr: np.ndarray) -> int:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	vert = cv2.flip(gray, 1)
	horz = cv2.flip(gray, 0)
	rot = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
	# Ensure same shape for similarity computation
	h, w = gray.shape[:2]
	vert = cv2.resize(vert, (w, h))
	horz = cv2.resize(horz, (w, h))
	rot = cv2.resize(rot, (w, h))
	def sim(a, b):
		na = (a - a.mean()) / (a.std() + 1e-6)
		nb = (b - b.mean()) / (b.std() + 1e-6)
		return float((na * nb).mean())
	axes = 0
	if sim(gray, vert) > 0.5:
		axes += 1
	if sim(gray, horz) > 0.5:
		axes += 1
	if sim(gray, rot) > 0.5:
		axes += 1
	return axes


def _detect_shapes(image_bgr: np.ndarray) -> List[str]:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 60, 180)
	contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	shapes = set()
	for cnt in contours:
		perim = cv2.arcLength(cnt, True)
		if perim < 40:
			continue
		approx = cv2.approxPolyDP(cnt, 0.03 * perim, True)
		n = len(approx)
		if n >= 8:
			shapes.add("circle")
		elif n == 3:
			shapes.add("triangle")
		elif n == 4:
			shapes.add("quad")
		else:
			shapes.add("arc")
	return sorted(list(shapes))


def _classify_type(dots_count: int, shapes: List[str]) -> str:
	if dots_count > 0 and "arc" in shapes:
		return "sikku"
	if dots_count > 0 and ("quad" in shapes or "triangle" in shapes):
		return "pulli"
	return "freehand" if dots_count == 0 else "unknown"


def _estimate_line_style(image_bgr: np.ndarray) -> str:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150)
	num_labels, labels = cv2.connectedComponents((edges > 0).astype(np.uint8))
	return "continuous" if num_labels <= 3 else "multiple"


def _estimate_stroke_order(dots: List[Tuple[int, int]]) -> List[str]:
	if not dots:
		return []
	sorted_dots = sorted(dots, key=lambda d: (d[1], d[0]))
	steps = []
	for i in range(len(sorted_dots) - 1):
		steps.append(f"dot{ i+1 }→dot{ i+2 } segment")
	return steps


def _downsample_polyline(points: List[Tuple[int, int]], step: int = 3) -> List[Tuple[int, int]]:
	if step <= 1 or len(points) <= 2:
		return points
	return [points[i] for i in range(0, len(points), step)] + ([points[-1]] if (len(points)-1) % step != 0 else [])


def _smooth_polyline(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
	n = len(points)
	if n < 7:
		return points
	polyorder = 2
	# choose the largest odd window <= n and > polyorder
	window = n if n % 2 == 1 else n - 1
	window = min(window, 21)  # cap to avoid over-smoothing long paths
	if window <= polyorder:
		window = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
	if window > n:
		window = n if n % 2 == 1 else n - 1
	if window <= polyorder or window < 3:
		return points
	x = np.array([p[0] for p in points], dtype=float)
	y = np.array([p[1] for p in points], dtype=float)
	try:
		xs = savgol_filter(x, window_length=int(window), polyorder=polyorder, mode="interp")
		ys = savgol_filter(y, window_length=int(window), polyorder=polyorder, mode="interp")
		return [(int(xs[i]), int(ys[i])) for i in range(n)]
	except Exception:
		return points

# New helpers for line extraction refinement

def _make_line_binary(image_bgr: np.ndarray) -> np.ndarray:
	# Assume white lines on dark bg. Convert to HSV and threshold by V and low saturation
	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	bright = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
	low_sat = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY_INV)[1]
	mask = cv2.bitwise_and(bright, low_sat)
	# fallback combine with Otsu on inverted gray
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	bw = cv2.bitwise_or(mask, otsu)
	return bw


def _remove_dots_from_bw(bw: np.ndarray, dots: List[Tuple[int, int]], avg_spacing: float) -> np.ndarray:
	clean = bw.copy()
	if not dots:
		return clean
	r = int(max(3, avg_spacing * 0.28))
	for (x, y) in dots:
		cv2.circle(clean, (int(x), int(y)), r, 0, thickness=-1)
	return clean


def _neighbors8(x: int, y: int, img: np.ndarray) -> List[Tuple[int, int]]:
	h, w = img.shape
	pts = []
	for dy in (-1, 0, 1):
		for dx in (-1, 0, 1):
			if dx == 0 and dy == 0:
				continue
			x2, y2 = x + dx, y + dy
			if 0 <= x2 < w and 0 <= y2 < h and img[y2, x2] > 0:
				pts.append((x2, y2))
	return pts


def _trace_skeleton_paths(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
	bin_img = (skel > 0).astype(np.uint8)
	points = np.argwhere(bin_img > 0)
	coords = {(int(x), int(y)) for y, x in points}
	def degree(pt: Tuple[int, int]) -> int:
		x, y = pt
		return sum(1 for n in _neighbors8(x, y, bin_img))
	endpoints = [p for p in coords if degree(p) == 1]
	visited: Set[Tuple[int, int]] = set()
	paths: List[List[Tuple[int, int]]] = []
	def walk(start: Tuple[int, int]):
		path = [start]
		visited.add(start)
		curr = start
		prev = None
		while True:
			nbrs = [n for n in _neighbors8(curr[0], curr[1], bin_img) if n != prev and n not in visited]
			if len(nbrs) == 0:
				break
			nxt = nbrs[0]
			if len(nbrs) > 1 and prev is not None:
				vx, vy = curr[0] - prev[0], curr[1] - prev[1]
				best = None
				best_dot = -1e9
				for c in nbrs:
					ux, uy = c[0] - curr[0], c[1] - curr[1]
					dot = vx * ux + vy * uy
					if dot > best_dot:
						best_dot = dot
						best = c
				nxt = best if len(nbrs) > 1 and prev is not None else nxt
			path.append(nxt)
			visited.add(nxt)
			prev, curr = curr, nxt
			# stop at junctions/endpoints
			deg = len(_neighbors8(curr[0], curr[1], bin_img))
			if deg == 1 or deg >= 3:
				break
		return path
	for ep in endpoints:
		if ep in visited:
			continue
		paths.append(walk(ep))
	# cover remaining cycles
	for p in list(coords):
		if p not in visited and bin_img[p[1], p[0]] > 0:
			paths.append(walk(p))
	return paths


def _prune_short_paths(paths: List[List[Tuple[int, int]]], min_len: int) -> List[List[Tuple[int, int]]]:
	return [p for p in paths if len(p) >= min_len]


def _stitch_paths(paths: List[List[Tuple[int, int]]], join_dist: int = 8) -> List[List[Tuple[int, int]]]:
	if not paths:
		return paths
	paths = [list(p) for p in paths]
	changed = True
	while changed:
		changed = False
		for i in range(len(paths)):
			if changed:
				break
			for j in range(i + 1, len(paths)):
				a = paths[i]
				b = paths[j]
				ends = [a[0], a[-1]]
				begs = [b[0], b[-1]]
				best = None
				best_d2 = 1e18
				for ei, ea in enumerate(ends):
					for bj, eb in enumerate(begs):
						d2 = (ea[0] - eb[0]) ** 2 + (ea[1] - eb[1]) ** 2
						if d2 < best_d2:
							best_d2 = d2
							best = (ei, bj)
				if best is not None and best_d2 <= join_dist * join_dist:
					ai, bj = best
					if ai == 0:
						a = list(reversed(a))
					if bj == 0:
						b = b
					else:
						b = list(reversed(b))
					paths[i] = a + b
					paths.pop(j)
					changed = True
					break
	return paths

# Legacy extraction: Canny + contours

def _extract_strokes_legacy(image_bgr: np.ndarray) -> List[Dict]:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	edges = cv2.Canny(gray, 60, 150)
	contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	strokes: List[Dict] = []
	for cnt in contours:
		if len(cnt) < 20:
			continue
		pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
		pts = _downsample_polyline(pts, step=4)
		strokes.append({"points": pts})
	return strokes

# Skeleton-based refined extraction

def _extract_strokes_skeleton(image_bgr: np.ndarray, dots: List[Tuple[int, int]], avg_spacing: float) -> List[Dict]:
	bw0 = _make_line_binary(image_bgr)
	bw = _remove_dots_from_bw(bw0, dots, avg_spacing)
	kernel = np.ones((3, 3), np.uint8)
	bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
	skel = skeletonize((bw > 0).astype(np.uint8)).astype(np.uint8) * 255
	paths = _trace_skeleton_paths(skel)
	paths = _prune_short_paths(paths, min_len=14)
	paths = _stitch_paths(paths, join_dist=int(max(6, avg_spacing * 0.3)))
	strokes: List[Dict] = []
	for pts in paths:
		pts = _downsample_polyline(pts, step=2)
		pts = _smooth_polyline(pts)
		if len(pts) >= 8:
			strokes.append({"points": pts})
	return strokes


def extract_kolam_features(image_bytes: bytes, stroke_mode: str = "legacy", *, dot_min_area: int = 12, dot_max_area: int = 6000, dot_hough_param2: int = 18, snap_grid: bool = True) -> Dict:
	image_bgr = _read_image_bytes(image_bytes)
	dots, avg_spacing = _detect_dots(image_bgr, min_area=dot_min_area, max_area=dot_max_area, hough_param2=dot_hough_param2)
	if snap_grid:
		dots = _snap_to_grid(dots)
	dots_count = len(dots)
	dot_grid_str, (rows, cols) = _estimate_grid(dots)
	sym_axes = _estimate_symmetry(image_bgr)
	shapes = _detect_shapes(image_bgr)
	kolam_type = _classify_type(dots_count, shapes)
	line_style = _estimate_line_style(image_bgr)
	stroke_order = _estimate_stroke_order(dots)
	strokes = _extract_strokes_legacy(image_bgr)

	data = {
		"kolam_type": kolam_type,
		"dots_count": int(dots_count),
		"dot_grid": dot_grid_str,
		"symmetry_axes": int(sym_axes),
		"shapes_detected": shapes,
		"line_style": line_style,
		"stroke_order": stroke_order,
		"measurements": {"avg_spacing": float(avg_spacing)},
		"dots": dots,
		"strokes": strokes,
		"render": {"stroke_color": "#0a6cf1", "stroke_width": 2},
	}
	return data
