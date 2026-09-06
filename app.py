# streamlit run app.py

import os
import io
import torch
import numpy as np
import rasterio
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import plotly.express as px
from streamlit_image_comparison import image_comparison
import mlstac
import tempfile
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    BASE_DIR
    / "SEN2SRLite_RGBN_x4"
    / "SEN2SRLite"
    / "NonReference_RGBN_x4"
)

# -------------------------------------------------------------------
# Configuration & CSS
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="VISTAARA", page_icon="🛰️", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stMetric, [data-testid="stMetric"], .stCode, [data-testid="stCodeBlock"], [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.9));
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 0 8px rgba(0, 255, 255, 0.2);
    }
    h1, h2, h3, h4 {
        color: #8b949e !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 400;
    }
    .vistaara-header {
        font-size: 3em;
        font-weight: 700;
        letter-spacing: 2px;
        color: #00ffff;
        margin-bottom: 0px;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.3);
    }
    .vistaara-sub {
        font-size: 1.2em;
        color: #8b949e;
        margin-bottom: 20px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
@st.cache_resource
def load_sen2srlite_model(device):
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Local model not found at {MODEL_DIR}")
    model = mlstac.load(MODEL_DIR).compiled_model(device=device)
    model.eval()
    return model

@st.cache_data
def load_and_preprocess_image(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        image = src.read()
        profile = src.profile.copy()
        
        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)
        pixel_area_m2 = pixel_width * pixel_height
        
        metadata = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs),
            "transform": src.transform,
            "pixel_area_m2": pixel_area_m2,
            "res": src.res
        }
    os.remove(tmp_path)
    return image, profile, metadata

def process_super_resolution(lr_tensor, model, device, scale=4, patch_size=128):
    channels, height, width = lr_tensor.shape
    sr_height = height * scale
    sr_width = width * scale
    sr_image = torch.zeros((channels, sr_height, sr_width), dtype=torch.float32)

    total_patches = len(range(0, height, patch_size)) * len(range(0, width, patch_size))
    progress_bar = st.progress(0, text="Processing patches...")
    patch_idx = 0

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch_height = min(patch_size, height - y)
            patch_width = min(patch_size, width - x)

            patch = lr_tensor[:, y:y+patch_height, x:x+patch_width]
            padded_patch = torch.zeros((4, patch_size, patch_size), dtype=torch.float32)
            padded_patch[:, :patch_height, :patch_width] = patch
            padded_patch = padded_patch.unsqueeze(0)

            with torch.no_grad():
                sr_patch = model(padded_patch.to(device))

            sr_patch = sr_patch[0].cpu()
            sr_patch = sr_patch[:, :patch_height*scale, :patch_width*scale]

            sr_y = y * scale
            sr_x = x * scale
            sr_image[:, sr_y:sr_y+sr_patch.shape[1], sr_x:sr_x+sr_patch.shape[2]] = sr_patch
            
            patch_idx += 1
            progress_bar.progress(patch_idx / total_patches, text=f"Processing patches... {patch_idx}/{total_patches}")

    progress_bar.empty()
    return sr_image.numpy()

def create_rgb_visualization(image_tensor, p2, p98):
    rgb = np.transpose(image_tensor[[0, 1, 2]], (1, 2, 0))
    # Apply shared stretch
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))

def calculate_scl_stats(scl, pixel_area_m2):
    # Mapping
    # 0 -> NoData (99)
    # 4 -> Vegetation (1)
    # 5 -> Non-Vegetation (2)
    # 6 -> Water (3)
    # Others -> Other (0)
    
    classified = np.zeros(scl.shape, dtype=np.uint8)
    classified[scl == 0] = 99
    classified[scl == 4] = 1
    classified[scl == 5] = 2
    classified[scl == 6] = 3
    
    # 1,2,3,7,8,9,10,11 are mapped to 0 (Other) as initialized
    
    total_valid_pixels = np.sum(classified != 99)
    nodata_pixels = np.sum(classified == 99)
    
    veg_px = np.sum(classified == 1)
    nonveg_px = np.sum(classified == 2)
    water_px = np.sum(classified == 3)
    other_px = np.sum(classified == 0)
    
    def get_pct(px):
        return (px / total_valid_pixels * 100) if total_valid_pixels > 0 else 0
        
    def get_area(px):
        return px * pixel_area_m2
        
    stats = {
        'total_valid_px': total_valid_pixels,
        'nodata_px': nodata_pixels,
        'classes': {
            'Vegetation': {'px': veg_px, 'pct': get_pct(veg_px), 'area_m2': get_area(veg_px)},
            'Non-Vegetation': {'px': nonveg_px, 'pct': get_pct(nonveg_px), 'area_m2': get_area(nonveg_px)},
            'Water': {'px': water_px, 'pct': get_pct(water_px), 'area_m2': get_area(water_px)},
            'Other': {'px': other_px, 'pct': get_pct(other_px), 'area_m2': get_area(other_px)}
        },
        'map': classified
    }
    return stats

# -------------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------------
st.sidebar.markdown("<h2 style='color:#00ffff;'>VISTAARA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Sentinel-2 Super-Resolution<br>& Geospatial Intelligence", unsafe_allow_html=True)
st.sidebar.markdown("---")

nav = st.sidebar.radio("Navigation", [
    "Overview", 
    "Resolution", 
    "Land Cover", 
    "Spectral Analysis", 
    "Area Analysis", 
    "Exports",
    "Change Detection (Before / After)"
])

# -------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------
st.markdown('<div class="vistaara-header">VISTAARA</div>', unsafe_allow_html=True)
st.markdown('<div class="vistaara-sub">Sentinel-2 Super-Resolution & Geospatial Intelligence</div>', unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

if nav == "Change Detection (Before / After)":
    st.subheader("Change Detection (Before / After)")
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Upload Before Image (Sentinel-2 GeoTIFF)", type=["tif", "tiff"])
    with col2:
        after_file = st.file_uploader("Upload After Image (Sentinel-2 GeoTIFF)", type=["tif", "tiff"])
    
    if before_file and after_file:
        try:
            with st.spinner("Processing Before Image..."):
                b_image, b_profile, b_meta = load_and_preprocess_image(before_file.getvalue())
                model = load_sen2srlite_model(device)
                b_lr_tensor = torch.from_numpy(b_image[[3, 2, 1, 7]].astype(np.float32))
                b_sr_image = process_super_resolution(b_lr_tensor, model, device)
                
            with st.spinner("Processing After Image..."):
                a_image, a_profile, a_meta = load_and_preprocess_image(after_file.getvalue())
                a_lr_tensor = torch.from_numpy(a_image[[3, 2, 1, 7]].astype(np.float32))
                a_sr_image = process_super_resolution(a_lr_tensor, model, device)
                
            with st.spinner("Computing Analysis..."):
                b_sr_rgb = b_sr_image[[0, 1, 2]]
                a_sr_rgb = a_sr_image[[0, 1, 2]]
                
                lr_rgb = np.transpose(b_image[[3, 2, 1]], (1, 2, 0))
                p2, p98 = np.percentile(lr_rgb, (2, 98))
                if p98 == p2: p98 = p2 + 1e-5
                
                b_pil = create_rgb_visualization(b_sr_image, p2, p98)
                a_pil = create_rgb_visualization(a_sr_image, p2, p98)
                
                st.markdown("### High-Resolution Preview (2.5m)")
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.image(b_pil, caption="Before Image (2.5m True Color)", use_container_width=True)
                with pc2:
                    st.image(a_pil, caption="After Image (2.5m True Color)", use_container_width=True)
                
                diff = np.abs(a_sr_rgb - b_sr_rgb)
                diff_norm = np.mean(diff, axis=0)
                diff_max = diff_norm.max()
                if diff_max > 0:
                    diff_norm = diff_norm / diff_max
                
                threshold = 0.18
                change_mask = diff_norm > threshold
                
                a_rgb_np = np.array(a_pil)
                highlight = np.zeros_like(a_rgb_np)
                highlight[change_mask] = [255, 20, 147]
                
                alpha = 0.5
                blended = a_rgb_np.copy()
                blended[change_mask] = (a_rgb_np[change_mask] * (1 - alpha) + highlight[change_mask] * alpha).astype(np.uint8)
                
                st.markdown("### Highlighted Change Map")
                st.image(Image.fromarray(blended), caption="Detected Changes (Neon Pink)", use_container_width=True)
                
                st.markdown("---")
                st.markdown("### Advanced Geospatial Analytics Engine")
                
                total_change_pixels = np.sum(change_mask)
                pixel_area_m2 = a_meta['pixel_area_m2'] / 16
                total_area_m2 = total_change_pixels * pixel_area_m2
                
                st.metric("Total Affected Footprint", f"{total_area_m2:,.2f} m²", f"{total_area_m2/1e6:,.4f} km²")
                
                st.markdown("#### Spatial Location & Coordinate Tracking")
                st.markdown("Main geographical coordinates of severe ground changes:")
                
                y_coords, x_coords = np.where(change_mask)
                if len(y_coords) > 0:
                    change_intensities = diff_norm[y_coords, x_coords]
                    sorted_indices = np.argsort(change_intensities)[::-1]
                    top_n = min(10, len(sorted_indices))
                    top_y = y_coords[sorted_indices[:top_n]]
                    top_x = x_coords[sorted_indices[:top_n]]
                    
                    scale = 4
                    new_transform = a_meta['transform'] * rasterio.Affine.scale(1/scale, 1/scale)
                    coords_list = []
                    for i in range(top_n):
                        lon, lat = new_transform * (top_x[i], top_y[i])
                        intensity = change_intensities[sorted_indices[i]] * 100
                        coords_list.append({"Latitude / Y": lat, "Longitude / X": lon, "Change Intensity (%)": f"{intensity:.1f}%"})
                    
                    st.table(pd.DataFrame(coords_list))
                else:
                    st.info("No significant changes detected.")
                    
        except Exception as e:
            st.error(f"Error processing change detection: {e}")

else:
    # 1. File Upload
    uploaded_file = st.sidebar.file_uploader("Upload Sentinel-2 GeoTIFF", type=["tif", "tiff"])
    
    if not uploaded_file:
        st.info("Please upload a Sentinel-2 GeoTIFF to begin analysis.")
        st.markdown("Status: <span class='vistaara-status'>● LOCAL AI ENGINE (Ready)</span>", unsafe_allow_html=True)
    else:
        # 2. Process Image (Cached)
        try:
            with st.spinner("Reading raster data..."):
                image, profile, metadata = load_and_preprocess_image(uploaded_file.getvalue())
        
            if metadata['count'] < 13:
                st.error(f"Invalid Sentinel-2 TIFF. Found {metadata['count']} bands, expected at least 13.")
                st.stop()
            
            if "processed" not in st.session_state or st.session_state.uploaded_name != uploaded_file.name:
                st.session_state.uploaded_name = uploaded_file.name
                st.session_state.image = image
                st.session_state.profile = profile
                st.session_state.metadata = metadata
            
                # SR Processing
                with st.spinner("Loading SEN2SRLite..."):
                    model = load_sen2srlite_model(device)
            
                lr_image = image[[3, 2, 1, 7]].astype(np.float32)
                lr_tensor = torch.from_numpy(lr_image)
            
                with st.spinner("Processing super-resolution..."):
                    sr_image = process_super_resolution(lr_tensor, model, device)
                
                with st.spinner("Integrating 2.5m high-resolution band stack..."):
                    import torch.nn.functional as F
                    full_img_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
                    full_img_upsampled = F.interpolate(full_img_tensor, scale_factor=4, mode='bilinear', align_corners=False)
                    full_img_upsampled = full_img_upsampled.squeeze(0).numpy()
                
                    # Overwrite upsampled bands with the sharp SR outputs
                    # lr_image mapping: 0: B04, 1: B03, 2: B02, 3: B08
                    full_img_upsampled[3] = sr_image[0]  # B04
                    full_img_upsampled[2] = sr_image[1]  # B03
                    full_img_upsampled[1] = sr_image[2]  # B02
                    full_img_upsampled[7] = sr_image[3]  # B08
                
                st.session_state.lr_image = lr_image
                st.session_state.sr_image = sr_image
                st.session_state.full_img_2_5m = full_img_upsampled
            
                # Shared Visualization Stretch (2nd to 98th percentile of Original RGB)
                lr_rgb = np.transpose(lr_image[[0, 1, 2]], (1, 2, 0))
                p2, p98 = np.percentile(lr_rgb, (2, 98))
            
                # Avoid division by zero
                if p98 == p2:
                    p98 = p2 + 1e-5
                
                st.session_state.p2 = p2
                st.session_state.p98 = p98
            
                st.session_state.lr_pil = create_rgb_visualization(lr_image, p2, p98)
                st.session_state.sr_pil = create_rgb_visualization(sr_image, p2, p98)
            
                # SCL Processing
                st.session_state.scl_stats = calculate_scl_stats(image[12], metadata['pixel_area_m2'])
                st.session_state.processed = True
            
            # Helper vars
            meta = st.session_state.metadata
            scl_stats = st.session_state.scl_stats
        
            # -------------------------------------------------------------------
            # Routing
            # -------------------------------------------------------------------
            if nav == "Overview":
                st.subheader("Scene Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Resolution", f"{meta['res'][0]:.1f} m → {meta['res'][0]/4:.1f} m")
                c2.metric("Total Area", f"{(meta['pixel_area_m2'] * meta['width'] * meta['height']) / 1e6:.2f} km²")
                c3.metric("Bands", f"{meta['count']}")
            
                st.markdown("---")
                st.markdown("### Model Information Panel")
                st.markdown("""
                **AI ENGINE:** SEN2SRLite RGBN ×4  
                **Input:** B04 / B03 / B02 / B08  
                **Input GSD:** 10 m  
                **Output:** 2.5 m pixel spacing  
                **Scale:** 4×  
                """)
            
                st.markdown("---")
                st.warning("""
                **SCIENTIFIC WARNING**  
                AI-enhanced imagery may contain reconstructed visual details. Enhanced pixels should not be treated as original sensor measurements.  
                For quantitative remote-sensing analysis, use the original Sentinel-2 spectral data whenever possible.
                """)
            
            elif nav == "Resolution":
                st.subheader("Original vs Enhanced Comparison")
                st.markdown("Compare the original 10 m pixel spacing with the High-Res Output (2.5m).")
            
                # The LR image tensor needs to be resized to match the SR shape so the slider feels responsive
                import torch.nn.functional as F
                lr_resized = F.interpolate(torch.from_numpy(st.session_state.lr_image).unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0).numpy()
                lr_resized_pil = create_rgb_visualization(lr_resized, st.session_state.p2, st.session_state.p98)
            
                image_comparison(
                    img1=st.session_state.sr_pil,
                    img2=lr_resized_pil,
                    label1="High-Res Output (2.5m)",
                    label2="Original Input (10m)",
                    starting_position=50,
                    make_responsive=True
                )
            
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### ORIGINAL")
                    st.markdown("Sentinel-2 · 10 m pixel spacing")
                    st.image(st.session_state.lr_pil, use_container_width=True)
                with c2:
                    st.markdown("#### High-Res Output (2.5m)")
                    st.markdown("SEN2SRLite RGBN ×4 · 2.5 m output pixel spacing")
                    st.image(st.session_state.sr_pil, use_container_width=True)
                
            elif nav == "Land Cover":
                st.subheader("SCL Classification")
            
                # Prepare color map
                cmap = np.zeros((meta['height'], meta['width'], 3), dtype=np.uint8)
                cmap[scl_stats['map'] == 1] = [34, 139, 34]   # Vegetation
                cmap[scl_stats['map'] == 2] = [210, 180, 140] # Non-Vegetation
                cmap[scl_stats['map'] == 3] = [0, 0, 255]     # Water
                cmap[scl_stats['map'] == 0] = [128, 128, 128] # Other
                cmap[scl_stats['map'] == 99] = [0, 0, 0]      # NoData
            
                st.session_state.scl_pil = Image.fromarray(cmap)
            
                # Upsample SCL map to match 2.5m canvas for clean comparison slider
                scl_resized = st.session_state.scl_pil.resize(st.session_state.sr_pil.size, Image.NEAREST)
            
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("Compare the High-Res Output (2.5m) with the SCL Classification map.")
                    image_comparison(
                        img1=st.session_state.sr_pil,
                        img2=scl_resized,
                        label1="High-Res Output (2.5m)",
                        label2="SCL Classification",
                        starting_position=50,
                        make_responsive=True
                    )
            
                with c2:
                    st.markdown("### Coverage Percentages")
                    st.markdown(f"🟢 **Vegetation:** {scl_stats['classes']['Vegetation']['pct']:.2f}%")
                    st.markdown(f"🟤 **Non-Vegetation:** {scl_stats['classes']['Non-Vegetation']['pct']:.2f}%")
                    st.markdown(f"🔵 **Water:** {scl_stats['classes']['Water']['pct']:.2f}%")
                    st.markdown(f"🔘 **Other:** {scl_stats['classes']['Other']['pct']:.2f}%")
                    if scl_stats['nodata_px'] > 0:
                        st.markdown(f"⚫ **NoData:** {scl_stats['nodata_px']} pixels (Excluded)")
                    
                st.markdown("---")
                st.markdown("### Detected Water Bodies")
                st.markdown(f"**Coverage:** {scl_stats['classes']['Water']['pct']:.2f}%")
                st.markdown(f"**Area:** {scl_stats['classes']['Water']['area_m2'] / 1e6:.2f} km²")
                st.info("Source: Sentinel-2 SCL (SCL-derived Water Classification)")
            
                st.markdown("### Vegetation Analysis")
                st.markdown(f"**Coverage:** {scl_stats['classes']['Vegetation']['pct']:.2f}%")
                st.markdown(f"**Area:** {scl_stats['classes']['Vegetation']['area_m2'] / 1e6:.2f} km²")
                st.info("Source: Sentinel-2 SCL (SCL-derived Vegetation Classification)")
            
                st.markdown("---")
                st.markdown("### ADVANCED SEMANTIC SEGMENTATION")
                st.markdown("Roads · Rivers · Lakes · Buildings · Trees · Land")
                st.warning("Status: Requires a compatible local segmentation model.")
            
            elif nav == "Spectral Analysis":
                st.subheader("Spectral Analysis")
                st.info("Calculated from the integrated High-Res 2.5m unified band stack.")
            
                view_option = st.selectbox("Select View", [
                    "True Color (B04/B03/B02)", 
                    "NDVI (B08-B04)/(B08+B04)", 
                    "NDWI (B03-B08)/(B03+B08)", 
                    "NDMI (B8A-B11)/(B8A+B11)"
                ])
            
                img = st.session_state.full_img_2_5m
            
                if view_option == "True Color (B04/B03/B02)":
                    st.image(st.session_state.sr_pil, caption="High-Res True Color (2.5m)", use_container_width=True)
                
                elif view_option == "NDVI (B08-B04)/(B08+B04)":
                    nir = img[7].astype(np.float32)
                    red = img[3].astype(np.float32)
                    denom = (nir + red)
                    denom[denom == 0] = 1e-5
                    ndvi = (nir - red) / denom
                
                    ndvi_ramp = [
                       (-1.0, '#0c0c0c'), (-0.2, '#0c0c0c'), (-0.1, '#bfbfbf'), (0.0, '#dbdbdb'),
                       (0.025, '#eaeaea'), (0.05, '#fff9cc'), (0.075, '#ede8b5'), (0.1, '#ddd89b'),
                       (0.125, '#ccc682'), (0.15, '#bcb76b'), (0.175, '#afc160'), (0.2, '#a3cc59'),
                       (0.25, '#91bf51'), (0.3, '#7fb247'), (0.35, '#70a33f'), (0.4, '#609635'),
                       (0.45, '#4f892d'), (0.5, '#3f7c23'), (0.55, '#306d1c'), (0.6, '#216011'),
                       (1.0, '#004400')
                    ]
                    vmin, vmax = -1.0, 1.0
                    colors = [((v - vmin) / (vmax - vmin), c) for v, c in ndvi_ramp]
                    cmap = mcolors.LinearSegmentedColormap.from_list('custom_ndvi', colors)
                
                    st.markdown(f"**NDVI Statistics (2.5m):** Min: {ndvi.min():.2f} | Max: {ndvi.max():.2f} | Mean: {ndvi.mean():.2f}")
                
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cax = ax.imshow(ndvi, cmap=cmap, vmin=vmin, vmax=vmax)
                    fig.colorbar(cax, ax=ax, orientation='vertical')
                    ax.axis('off')
                    st.pyplot(fig)
                
                elif view_option == "NDWI (B03-B08)/(B03+B08)":
                    green = img[2].astype(np.float32)
                    nir = img[7].astype(np.float32)
                    denom = (green + nir)
                    denom[denom == 0] = 1e-5
                    ndwi = (green - nir) / denom
                
                    ndwi_ramp = [(-0.8, '#008000'), (0.0, '#FFFFFF'), (0.8, '#0000CC')]
                    vmin, vmax = -0.8, 0.8
                    colors = [((v - vmin) / (vmax - vmin), c) for v, c in ndwi_ramp]
                    cmap = mcolors.LinearSegmentedColormap.from_list('custom_ndwi', colors)
                
                    st.markdown(f"**NDWI Statistics (2.5m):** Min: {ndwi.min():.2f} | Max: {ndwi.max():.2f} | Mean: {ndwi.mean():.2f}")
                
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cax = ax.imshow(ndwi, cmap=cmap, vmin=vmin, vmax=vmax)
                    fig.colorbar(cax, ax=ax, orientation='vertical')
                    ax.axis('off')
                    st.pyplot(fig)
                
                elif view_option == "NDMI (B8A-B11)/(B8A+B11)":
                    b8a = img[8].astype(np.float32) if meta['count'] > 8 else img[7].astype(np.float32)
                    b11 = img[11].astype(np.float32) if meta['count'] > 11 else img[7].astype(np.float32)
                    denom = (b8a + b11)
                    denom[denom == 0] = 1e-5
                    ndmi = (b8a - b11) / denom
                
                    ndmi_ramp = [
                       (-0.8, '#800000'), (-0.24, '#ff0000'), (-0.032, '#ffff00'),
                       (0.032, '#00ffff'), (0.24, '#0000ff'), (0.8, '#000080')
                    ]
                    vmin, vmax = -0.8, 0.8
                    colors = [((v - vmin) / (vmax - vmin), c) for v, c in ndmi_ramp]
                    cmap = mcolors.LinearSegmentedColormap.from_list('custom_ndmi', colors)
                
                    st.markdown(f"**NDMI Statistics (2.5m):** Min: {ndmi.min():.2f} | Max: {ndmi.max():.2f} | Mean: {ndmi.mean():.2f}")
                
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cax = ax.imshow(ndmi, cmap=cmap, vmin=vmin, vmax=vmax)
                    fig.colorbar(cax, ax=ax, orientation='vertical')
                    ax.axis('off')
                    st.pyplot(fig)

                
            elif nav == "Area Analysis":
                st.subheader("Geographic Area Analysis")
            
                classes = scl_stats['classes']
            
                m1, m2, m3 = st.columns(3)
                m1.metric("TOTAL AREA", f"{(scl_stats['total_valid_px'] * meta['pixel_area_m2'])/1e6:.2f} km²")
                m2.metric("VEGETATION", f"{classes['Vegetation']['area_m2']/1e6:.2f} km²")
                m3.metric("NON-VEGETATION", f"{classes['Non-Vegetation']['area_m2']/1e6:.2f} km²")
            
                m4, m5 = st.columns(2)
                m4.metric("WATER", f"{classes['Water']['area_m2']/1e6:.2f} km²")
                m5.metric("OTHER", f"{classes['Other']['area_m2']/1e6:.2f} km²")
            
                st.markdown("---")
            
                # Charts
                labels = list(classes.keys())
                values = [c['px'] for c in classes.values()]
                colors = ['#228B22', '#D2B48C', '#0000FF', '#808080']
            
                c1, c2 = st.columns(2)
                fig1 = px.pie(names=labels, values=values, title='Land Cover Distribution',
                             color=labels, color_discrete_map={k: v for k, v in zip(labels, colors)}, hole=0.4)
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
                c1.plotly_chart(fig1, use_container_width=True)
            
                fig2 = px.bar(x=labels, y=[c['area_m2']/1e6 for c in classes.values()], title='Area Distribution (km²)',
                              labels={'x': 'Class', 'y': 'Area (km²)'},
                              color=labels, color_discrete_map={k: v for k, v in zip(labels, colors)})
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
                c2.plotly_chart(fig2, use_container_width=True)
            
                # Table
                st.markdown("### Area & Percentage Table")
                df = pd.DataFrame([
                    {
                        'Class': k,
                        'Pixel Count': v['px'],
                        'Percentage': f"{v['pct']:.2f}%",
                        'Area m²': f"{v['area_m2']:,.1f}",
                        'Area ha': f"{v['area_m2']/10000:,.2f}",
                        'Area km²': f"{v['area_m2']/1e6:,.2f}"
                    } for k, v in classes.items()
                ])
                st.dataframe(df, use_container_width=True)
            

            
            elif nav == "Exports":
                st.subheader("Download Outputs")
                st.markdown("Export the generated AI-reconstructed visualizations and SCL metrics.")
            
                c1, c2, c3 = st.columns(3)
            
                # RGB PNG
                buf = io.BytesIO()
                st.session_state.sr_pil.save(buf, format="PNG")
                c1.download_button("Download Enhanced RGB PNG", data=buf.getvalue(), 
                                   file_name=f"{st.session_state.uploaded_name.split('.')[0]}_vistaara_sr.png", 
                                   mime="image/png")
                               
                # SCL PNG
                if 'scl_pil' in st.session_state:
                    buf2 = io.BytesIO()
                    st.session_state.scl_pil.save(buf2, format="PNG")
                    c2.download_button("Download SCL Classification PNG", data=buf2.getvalue(), 
                                       file_name=f"{st.session_state.uploaded_name.split('.')[0]}_scl_classification.png", 
                                       mime="image/png")
                                   
                # Enhanced TIF
                scale = 4
                new_transform = meta['transform'] * rasterio.Affine.scale(1/scale, 1/scale)
                new_profile = st.session_state.profile.copy()
                new_profile.update({
                    'height': st.session_state.sr_image.shape[1],
                    'width': st.session_state.sr_image.shape[2],
                    'count': 4,
                    'dtype': 'float32',
                    'transform': new_transform
                })
            
                buf3 = io.BytesIO()
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(**new_profile) as dataset:
                        dataset.write(st.session_state.sr_image)
                    buf3.write(memfile.read())
            
                c3.download_button("Download Enhanced GeoTIFF", data=buf3.getvalue(), 
                                   file_name=f"{st.session_state.uploaded_name.split('.')[0]}_vistaara_sr.tif", 
                                   mime="image/tiff")
                               
                # CSV Statistics
                classes = scl_stats['classes']
                df = pd.DataFrame([
                    {
                        'Class': k,
                        'Pixel Count': v['px'],
                        'Percentage': v['pct'],
                        'Area m²': v['area_m2'],
                        'Area hectares': v['area_m2']/10000,
                        'Area km²': v['area_m2']/1e6
                    } for k, v in classes.items() 
                ])
                csv = df.to_csv(index=False)
                st.download_button("Download CSV Statistics", data=csv, 
                                   file_name=f"{st.session_state.uploaded_name.split('.')[0]}_statistics.csv", 
                                   mime="text/csv")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
