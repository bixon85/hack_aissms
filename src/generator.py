from typing import Dict, List, Tuple
import numpy as np
import random


def _make_grid(grid_size: Tuple[int, int], grid_kind: str, spacing: int = 20) -> List[Tuple[int, int]]:
	rows, cols = grid_size
	points = []
	if grid_kind in ("square", "rectangular"):
		for r in range(rows):
			for c in range(cols):
				points.append((c * spacing, r * spacing))
	elif grid_kind == "triangular":
		for r in range(rows):
			for c in range(r + 1):
				points.append((c * spacing + (rows - r) * spacing * 0.5, r * spacing))
	else:
		for r in range(rows):
			for c in range(cols):
				points.append((c * spacing, r * spacing))
	if points:
		xs = [p[0] for p in points]
		ys = [p[1] for p in points]
		x0 = (max(xs) + min(xs)) / 2
		y0 = (max(ys) + min(ys)) / 2
		points = [(int(p[0] - x0), int(p[1] - y0)) for p in points]
	return points


def _wrap_strokes_orthogonal(dots: List[Tuple[int, int]], spacing: int, offset_factor: float) -> List[Dict]:
	strokes = []
	offset = int(max(4, spacing * offset_factor))
	for (x, y) in dots:
		pts = [(x - offset, y), (x, y - offset), (x + offset, y), (x, y + offset), (x - offset, y)]
		strokes.append({"points": pts})
	return strokes


def _connect_grid_lines(dots: List[Tuple[int, int]]) -> List[Dict]:
	if not dots:
		return []
	sorted_d = sorted(dots, key=lambda p: (p[1], p[0]))
	rows = []
	row = [sorted_d[0]]
	for i in range(1, len(sorted_d)):
		if abs(sorted_d[i][1] - row[-1][1]) <= 10:
			row.append(sorted_d[i])
		else:
			rows.append(row)
			row = [sorted_d[i]]
	rows.append(row)
	strokes = []
	for r in rows:
		if len(r) >= 2:
			strokes.append({"points": r})
	return strokes


def _connect_all_dots_path(dots: List[Tuple[int, int]]) -> List[Dict]:
	if not dots:
		return []
	remaining = dots[:]
	path = [remaining.pop(0)]
	while remaining:
		last = path[-1]
		next_idx = min(range(len(remaining)), key=lambda i: (remaining[i][0]-last[0])**2 + (remaining[i][1]-last[1])**2)
		path.append(remaining.pop(next_idx))
	return [{"points": path}]


def _apply_symmetry(strokes: List[Dict], symmetry: str, radial_order: int = 0) -> List[Dict]:
	if symmetry == "none" and radial_order <= 1:
		return strokes
	aug = list(strokes)
	for s in strokes:
		pts = s["points"]
		if symmetry in ("vertical", "radial"):
			aug.append({"points": [(-x, y) for (x, y) in pts]})
		if symmetry in ("horizontal", "radial"):
			aug.append({"points": [(x, -y) for (x, y) in pts]})
		if radial_order and radial_order > 1:
			# rotate around origin
			rads = []
			for k in range(1, radial_order):
				angle = 2 * np.pi * k / radial_order
				cosA, sinA = np.cos(angle), np.sin(angle)
				rads.append({"points": [(int(x * cosA - y * sinA), int(x * sinA + y * cosA)) for (x, y) in pts]})
			aug.extend(rads)
	return aug


def _chaikin(points: List[Tuple[int, int]], rounds: int) -> List[Tuple[int, int]]:
	if rounds <= 0 or len(points) < 3:
		return points
	pts = [(float(x), float(y)) for (x, y) in points]
	for _ in range(rounds):
		next_pts = [pts[0]]
		for i in range(len(pts) - 1):
			p = pts[i]
			q = pts[i + 1]
			Q = (0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1])
			R = (0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1])
			next_pts.extend([Q, R])
		next_pts.append(pts[-1])
		pts = next_pts
	return [(int(x), int(y)) for (x, y) in pts]


def generate_kolam(
	grid_size=(4, 4),
	grid_kind="square",
	pattern_type="sikku",
	symmetry="none",
	spacing: int = 20,
	seed: int = 0,
	connect_all: bool = False,
	stroke_color: str = "#0a6cf1",
	stroke_width: int = 2,
	wrap_offset_factor: float = 0.35,
	radial_symmetry_order: int = 0,
	smoothing_rounds: int = 0,
	stroke_passes: int = 1,
) -> Dict:
	random.seed(seed)
	dots = _make_grid(grid_size, grid_kind, spacing)
	# Base strokes
	if connect_all:
		strokes = _connect_all_dots_path(dots)
	elif pattern_type in ("sikku", "square"):
		strokes = _wrap_strokes_orthogonal(dots, spacing, wrap_offset_factor)
	elif pattern_type in ("pulli", "triangular"):
		strokes = _connect_grid_lines(dots)
	else:
		strokes = _connect_grid_lines(dots)
	# Optional: duplicate passes
	strokes = strokes * max(1, int(stroke_passes))
	# Symmetry
	strokes = _apply_symmetry(strokes, symmetry, radial_order=int(radial_symmetry_order))
	# Smoothing
	if smoothing_rounds and smoothing_rounds > 0:
		for s in strokes:
			pts = s.get("points", [])
			if len(pts) >= 3:
				s["points"] = _chaikin(pts, rounds=int(smoothing_rounds))

	data = {
		"kolam_type": pattern_type,
		"dots_count": len(dots),
		"dot_grid": f"{grid_size[0]}x{grid_size[1]}",
		"symmetry_axes": 0 if symmetry == "none" else (2 if symmetry in ("vertical", "horizontal") else 4),
		"shapes_detected": ["arc"],
		"line_style": "continuous",
		"stroke_order": [],
		"measurements": {"avg_spacing": spacing},
		"dots": dots,
		"strokes": strokes,
		"render": {"stroke_color": stroke_color, "stroke_width": int(stroke_width)},
	}
	return data
