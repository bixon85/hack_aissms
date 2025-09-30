from typing import Dict, List, Tuple, Optional
import io

import matplotlib.pyplot as plt


def _draw_dots(ax, dots: List[Tuple[int, int]]):
	if not dots:
		return
	xs = [d[0] for d in dots]
	ys = [d[1] for d in dots]
	ax.scatter(xs, ys, s=20, c="#333333")


def _draw_grid_axes(ax, dots):
	if not dots:
		ax.axis("off")
		return
	xs = [d[0] for d in dots]
	ys = [d[1] for d in dots]
	ax.set_xlim(min(xs) - 40, max(xs) + 40)
	ax.set_ylim(max(ys) + 40, min(ys) - 40)
	ax.set_aspect("equal")
	ax.axis("off")


def _draw_strokes(ax, data: Dict):
	strokes = data.get("strokes", [])
	render = data.get("render", {})
	color = render.get("stroke_color", "#0a6cf1")
	width = render.get("stroke_width", 2)
	for stroke in strokes:
		pts = stroke.get("points", [])
		if len(pts) < 2:
			continue
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		ax.plot(xs, ys, color=color, linewidth=width)


def draw_kolam_from_json(data: Dict, save_path: Optional[str] = None):
	fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
	dots = data.get("dots", [])
	_draw_dots(ax, dots)
	_draw_strokes(ax, data)
	_draw_grid_axes(ax, dots)

	buf = io.BytesIO()
	plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
	png_bytes = buf.getvalue()
	buf.close()

	if save_path:
		with open(save_path, "wb") as f:
			f.write(png_bytes)

	return fig, png_bytes
