import io
import json
from datetime import datetime

import numpy as np
import streamlit as st
import pandas as pd

from src.extractor import extract_kolam_features
from src.renderer import draw_kolam_from_json
from src.generator import generate_kolam

st.set_page_config(page_title="Kolam Image-to-Pattern & Generator", layout="wide")

st.title("Kolam Image-to-Pattern & Generator")

TAB_FROM_IMAGE, TAB_GENERATE = st.tabs(["From Image", "Generate New"]) 

@st.cache_data(show_spinner=False)
def _cached_extract(file_bytes: bytes, min_area: int, max_area: int, hough_param2: int, snap_grid: bool):
	return extract_kolam_features(
		file_bytes,
		stroke_mode="legacy",
		dot_min_area=int(min_area),
		dot_max_area=int(max_area),
		dot_hough_param2=int(hough_param2),
		snap_grid=bool(snap_grid),
	)

@st.cache_data(show_spinner=False)
def _cached_generate(params_json: str):
	params = json.loads(params_json)
	return generate_kolam(**params)

with TAB_FROM_IMAGE:
	left_col, right_col = st.columns([1, 1])

	with left_col:
		uploaded = st.file_uploader("Upload a Kolam image (PNG/JPG)", type=["png", "jpg", "jpeg"])
		with st.expander("Dot detection tuning"):
			dot_min_area = st.slider("Min dot area", 5, 80, 12)
			dot_max_area = st.slider("Max dot area", 100, 10000, 6000)
			dot_hough_param2 = st.slider("Hough sensitivity (lower = more)", 10, 50, 18)
			snap_grid = st.checkbox("Snap dots to nearest grid bands", value=True)

		features = None
		if uploaded is not None:
			file_bytes = uploaded.read()
			features = _cached_extract(file_bytes, dot_min_area, dot_max_area, dot_hough_param2, snap_grid)
			st.subheader("Extracted JSON Data")
			st.json(features)
			st.download_button(
				label="Download JSON",
				data=json.dumps(features, indent=2),
				file_name=f"kolam_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
				mime="application/json",
			)
			if features.get("dots"):
				df = pd.DataFrame(features["dots"], columns=["x", "y"])
				csv = df.to_csv(index=False).encode("utf-8")
				st.download_button("Download Dots CSV", csv, file_name="dots.csv", mime="text/csv")

	with right_col:
		if uploaded is not None and features is not None:
			fig, png_bytes = draw_kolam_from_json(features)
			st.subheader("Recreated Kolam")
			st.pyplot(fig, clear_figure=True)
			st.download_button(
				label="Download PNG",
				data=png_bytes,
				file_name=f"kolam_recreated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
				mime="image/png",
			)
			if st.button("Generate Drawing Animation"):
				from PIL import Image
				frames: list[Image.Image] = []
				partial = {**features, "strokes": []}
				for s in features.get("strokes", [])[:150]:
					partial["strokes"].append(s)
					f, bytes_png = draw_kolam_from_json(partial)
					frames.append(Image.open(io.BytesIO(bytes_png)).convert("RGBA"))
				if frames:
					buf = io.BytesIO()
					frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=120, loop=0)
					gif_bytes = buf.getvalue()
					st.image(gif_bytes)
					st.download_button("Download Animation GIF", gif_bytes, file_name="kolam_redraw.gif", mime="image/gif")

	st.markdown("""
	**How it works**:
	- Uses contour-based extraction for stable legacy behavior
	- Produces JSON and a clean redraw for study
	- Optional GIF shows stroke-by-stroke buildup
	""")

with TAB_GENERATE:
	colA, colB = st.columns([1, 1])
	with colA:
		st.subheader("Parameters")
		grid_kind = st.selectbox("Grid type", ["square", "rectangular", "triangular"], index=0)
		if grid_kind == "square":
			g = st.slider("Grid size (NxN)", 2, 16, 5)
			grid_size = (g, g)
		elif grid_kind == "rectangular":
			rows = st.slider("Rows", 2, 20, 6)
			cols = st.slider("Cols", 2, 20, 8)
			grid_size = (rows, cols)
		else:
			g = st.slider("Triangular N (rows)", 2, 16, 6)
			grid_size = (g, g)

		pattern_type = st.selectbox("Pattern type", ["sikku", "pulli", "freehand", "square", "triangular", "star"]) 
		symmetry = st.selectbox("Symmetry", ["none", "vertical", "horizontal", "radial"]) 
		spacing = st.slider("Dot spacing (px)", 10, 60, 24)
		seed = st.number_input("Random seed (optional)", value=0, step=1)

		with st.expander("Advanced strokes"):
			connect_all = st.checkbox("Connect all dots in one continuous stroke", value=False)
			wrap_offset_factor = st.slider("Wrap offset factor (sikku)", 0.1, 0.6, 0.35)
			radial_order = st.slider("Radial symmetry order", 0, 12, 0, help="0 disables radial repetition; >1 adds rotation copies")
			smoothing_rounds = st.slider("Smoothing rounds (Chaikin)", 0, 4, 0)
			stroke_passes = st.slider("Duplicate stroke passes", 1, 3, 1)

		with st.expander("Style"):
			stroke_color = st.color_picker("Stroke color", value="#0a6cf1")
			stroke_width = st.slider("Stroke width", 1, 8, 2)

		# Build params dict and cache key
		gen_params = {
			"grid_size": grid_size,
			"grid_kind": grid_kind,
			"pattern_type": pattern_type,
			"symmetry": symmetry,
			"spacing": int(spacing),
			"seed": int(seed),
			"connect_all": bool(connect_all),
			"stroke_color": stroke_color,
			"stroke_width": int(stroke_width),
			"wrap_offset_factor": float(wrap_offset_factor),
			"radial_symmetry_order": int(radial_order),
			"smoothing_rounds": int(smoothing_rounds),
			"stroke_passes": int(stroke_passes),
		}
		params_json = json.dumps(gen_params, sort_keys=True)

	with colB:
		data = _cached_generate(params_json)
		fig, png_bytes = draw_kolam_from_json(data)
		st.subheader("Generated Kolam")
		st.pyplot(fig, clear_figure=True)
		st.download_button(
			label="Download JSON",
			data=json.dumps(data, indent=2),
			file_name=f"kolam_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
			mime="application/json",
		)
		st.download_button(
			label="Download PNG",
			data=png_bytes,
			file_name=f"kolam_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
			mime="image/png",
		)
		if data.get("dots"):
			df = pd.DataFrame(data["dots"], columns=["x", "y"])
			csv = df.to_csv(index=False).encode("utf-8")
			st.download_button("Download Dots CSV", csv, file_name="generated_dots.csv", mime="text/csv")

st.divider()

with st.expander("Educational: Geometry, Recursion, Symmetry"):
	st.markdown(
		"""
		- Kolam often use reflective and rotational symmetry to reduce complexity while preserving aesthetics.
		- Recursion arises from repeating drawing rules to cover the grid without crossings.
		- Generated strokes follow grid-adjacent offsets to simulate sikku wrapping patterns.
		- The connect-all option constructs a greedy near-Hamiltonian tour over dots.
		"""
	)
