import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import PatientAnalyzer
from environment import CancerSimulation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import time

st.set_page_config(page_title="The Peacekeeper: AI Immunotherapy", layout="wide", initial_sidebar_state="expanded")

# --- 1. HELPER FUNCTION: STATIC PLOTLY FALLBACK ---
def create_tumor_visualization(tumor_size, res_level_or_list, max_res=15.0):
    """Fallback Plotly visualization for the static before/after panels below."""
    if 'cell_coordinates' not in st.session_state:
        st.session_state.cell_coordinates = np.random.rand(20000, 2)

    num_cells = int(min(len(st.session_state.cell_coordinates), max(1, tumor_size)))
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()

    if isinstance(res_level_or_list, (list, np.ndarray)):
        cell_resistances = np.array(res_level_or_list[:num_cells])
        if len(cell_resistances) < num_cells:
            avg_res = np.mean(cell_resistances) if len(cell_resistances) > 0 else max_res / 2
            cell_resistances = np.pad(cell_resistances, (0, num_cells - len(cell_resistances)), constant_values=avg_res)
        avg_resistance = np.mean(cell_resistances)
    else:
        cell_resistances = np.full(num_cells, res_level_or_list)
        avg_resistance = res_level_or_list

    norm_resistances = np.clip(cell_resistances / max_res, 0, 1)

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=cell_coords[:, 0],
                y=cell_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=6, color=norm_resistances, colorscale='YlOrRd',
                    cmin=0, cmax=1, showscale=True,
                    colorbar=dict(title='Resistance', thickness=12, len=0.65, y=0.5, ticks='outside', tickfont=dict(color='#C9D1D9')),
                    opacity=0.85, line=dict(width=0)
                ),
                hovertemplate='Resistance: %{marker.color:.2f}<extra></extra>'
            )
        ]
    )

    fig.update_layout(
        template=None, paper_bgcolor='#0E1117', plot_bgcolor='#161B22', margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1], scaleanchor='x'),
        title=dict(text=f'Tumor Cells: {num_cells:,} | Avg Resistance: {avg_resistance:.1f}', x=0.5, font=dict(color='#C9D1D9', size=14)),
    )
    return fig

# --- 2. HELPER FUNCTION: CANVAS PHYSICS SIMULATOR ---
def canvas_physics_visualization(target_size, resistance_level, max_res=15.0):
    """
    Renders the Streamlit Drawable Canvas animated with physics.
    """
    # Resistance color mapping
    norm_res = min(max(resistance_level / max_res, 0), 1)
    r = int(255)
    g = int(255 - (255 * norm_res))
    b = int(178 - (178 * norm_res))
    color_hex = f"#{r:02x}{g:02x}{b:02x}"

    # Scale dots so the canvas doesn't crash from too many JSON objects
    vis_size = max(1, min(250, int(target_size / 4)))
    
    if "anim_cells" not in st.session_state:
        st.session_state.anim_cells = [{"x": 350, "y": 200, "vx": 0, "vy": 0, "r": 6} for _ in range(vis_size)]
        st.session_state.last_vis_size = vis_size

    FRICTION = 0.85
    SPLIT_FORCE = 15
    
    current_cells = st.session_state.anim_cells
    current_size = len(current_cells)
    
    # TRIGGER: When the user clicked Next/Prev, the target size changes
    if vis_size != st.session_state.last_vis_size:
        if vis_size > current_size:
            # DUPLICATE: Tumor Grew
            diff = vis_size - current_size
            for _ in range(diff):
                parent = current_cells[np.random.randint(0, max(1, current_size))] if current_size > 0 else {"x": 350, "y": 200}
                angle = np.random.uniform(0, 2 * np.pi)
                force = np.random.uniform(SPLIT_FORCE * 0.5, SPLIT_FORCE)
                current_cells.append({
                    "x": parent.get("x", 350), 
                    "y": parent.get("y", 200),
                    "vx": np.cos(angle) * force,
                    "vy": np.sin(angle) * force,
                    "r": 6
                })
        elif vis_size < current_size:
            # SHRINK: Tumor died off
            current_cells = current_cells[:vis_size]
        
        st.session_state.last_vis_size = vis_size

    # APPLY PHYSICS
    is_moving = False
    for cell in current_cells:
        cell["x"] += cell["vx"]
        cell["y"] += cell["vy"]
        cell["vx"] *= FRICTION
        cell["vy"] *= FRICTION
        
        # Keep inside canvas bounds (700x400)
        cell["x"] = max(10, min(690, cell["x"]))
        cell["y"] = max(10, min(390, cell["y"]))
        
        # If any cell is still moving fast enough, keep the "game loop" running
        if abs(cell["vx"]) > 0.5 or abs(cell["vy"]) > 0.5:
            is_moving = True
            
    st.session_state.anim_cells = current_cells

    # Convert state to Canvas JSON format
    objects = []
    for cell in current_cells:
        objects.append({
            "type": "circle",
            "left": cell["x"],
            "top": cell["y"],
            "radius": cell["r"],
            "fill": color_hex,
            "stroke": "white",
            "strokeWidth": 1,
            "originX": "center",
            "originY": "center"
        })
        
    drawing_data = {"version": "4.4.0", "objects": objects}
    
    st.markdown(f"<div style='text-align: center; color: #C9D1D9; font-family: sans-serif; padding-bottom: 5px; font-weight: bold;'>Dynamic Evolution | Target Cells: {target_size} | Resistance: {resistance_level:.2f}</div>", unsafe_allow_html=True)
    
    col_spacer1, col_canvas, col_spacer2 = st.columns([1, 4, 1])
    with col_canvas:
        st_canvas(
            initial_drawing=drawing_data,
            drawing_mode="transform",
            display_toolbar=False,
            update_streamlit=False,
            height=400,
            width=700,
            background_color="#161B22",
            key="physics_canvas" # Stays identical so the canvas replaces itself
        )
    
    # THE "GAME LOOP"
    if is_moving:
        time.sleep(0.05)
        st.rerun()

# --- 3. CONFIGURATION & MODEL LOAD ---
MODEL_PATH = "peacekeeper_final_azure"

@st.cache_resource
def load_model():
    if os.path.exists(f"{MODEL_PATH}.zip"):
        return PPO.load(MODEL_PATH)
    else:
        st.error(f"Model file {MODEL_PATH}.zip not found!")
        return None

st.title("🛡️ The Peacekeeper")
st.markdown("### Digital Immunotherapy & Evolutionary Trap Optimizer")

# --- 4. SIDEBAR: PATIENT DATA ---
st.sidebar.header("Patient Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Proteomic CSV", type=["csv"])

if 'treatment_history' not in st.session_state: st.session_state.treatment_history = None
if 'current_phase_step' not in st.session_state: st.session_state.current_phase_step = 0
if 'current_file_name' not in st.session_state: st.session_state.current_file_name = None

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.current_file_name:
        st.session_state.treatment_history = None
        st.session_state.current_phase_step = 0
        st.session_state.current_file_name = uploaded_file.name
        if "anim_cells" in st.session_state: del st.session_state.anim_cells
    
    data = pd.read_csv(uploaded_file)
    model = load_model()
    
    try:
        analyzer = PatientAnalyzer(df=data)
        profile = analyzer.get_patient_profile(data)
        cell_resistance = analyzer.get_cell_resistance_data()
        st.session_state.cell_resistance_data = cell_resistance
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Resistance (Drug A)", f"{profile['max_res_a']:.2f}")
    col2.metric("Avg Growth Rate", f"{profile['avg_growth']:.2f}%")
    col3.metric("Initial Res_B", "5.00 (Standard)")

    if st.button("Generate Optimized Treatment Plan"):
        env = CancerSimulation(profile)
        obs, _ = env.reset()
        history = []
        terminated = False
        day = 1
        
        while not terminated and day <= 30:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            action_name = "Rest (Recovery)" if action == 0 else "Drug A (Priming)" if action == 1 else "Drug B (TRAP)"
            size, res_a, res_b = obs[0], obs[1], obs[2]
            status = "🟢 SAFE" if info.get('toxicity', day) < 5 else "🟡 MONITOR"
            
            history.append({
                "Day": day,
                "Action": action_name,
                "Tumor Size": int(size),
                "Resist_A": round(res_a, 2),
                "Resist_B": round(res_b, 2),
                "Safety Status": status
            })
            if size <= 0: break
            day += 1
        
        st.session_state.treatment_history = history
        st.session_state.current_phase_step = 0
        if "anim_cells" in st.session_state: del st.session_state.anim_cells
    
    # --- 5. VISUALIZATION DISPLAY ---
    if st.session_state.treatment_history is not None:
        history = st.session_state.treatment_history
        max_steps = len(history) * 2 - 1 
        
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        
        # Step Navigation
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])
        
        with nav_col1:
            if st.button("◀ PREV", key="prev_day"):
                if st.session_state.current_phase_step > 0:
                    st.session_state.current_phase_step -= 1
                    st.rerun()
        
        current_idx = st.session_state.current_phase_step // 2
        is_after_treatment = (st.session_state.current_phase_step % 2 != 0)
        current_day_data = history[current_idx]
        
        phase_text = "🟢 After Treatment" if is_after_treatment else "🔴 Before Treatment"
        
        with nav_col3:
            st.markdown(f"<div style='text-align: center; padding: 10px; margin-top: 8px;'><h3>📅 Day {current_day_data['Day']} - {phase_text}</h3></div>", unsafe_allow_html=True)
        
        with nav_col5:
            if st.button("NEXT ▶", key="next_day"):
                if st.session_state.current_phase_step < max_steps:
                    st.session_state.current_phase_step += 1
                    st.rerun()
        
        # Calculate Target Sizing based on "Before" / "After"
        base_size = current_day_data["Tumor Size"]
        res_level = current_day_data["Resist_A"]
        
        if st.session_state.current_phase_step == 0:
            target_size = int(base_size * 1.15)
        else:
            if not is_after_treatment:
                target_size = int(base_size * 1.15) # Growth overnight
            else:
                target_size = base_size # After drugs applied

        # --- MAIN DISPLAY: PHYSICS CANVAS ---
        st.markdown("---")
        canvas_physics_visualization(target_size, res_level)
        
        # STATIC BEFORE/AFTER PANELS
        st.markdown("---")
        toggle_col1, toggle_col2 = st.columns([1, 1])
        with toggle_col1:
            show_before = st.toggle("🔴 Show Static Before", value=False)
        with toggle_col2:
            show_after = st.toggle("🟢 Show Static After", value=False)
        
        if show_before or show_after:
            vis_col1, vis_col2 = st.columns(2)
            cell_resistance = st.session_state.cell_resistance_data or res_level
            
            if show_before:
                with vis_col1:
                    st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
                    st.plotly_chart(create_tumor_visualization(int(current_day_data["Tumor Size"] * 1.15), cell_resistance), use_container_width=True)
            if show_after:
                with vis_col2:
                    st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
                    st.plotly_chart(create_tumor_visualization(current_day_data["Tumor Size"], cell_resistance), use_container_width=True)
        
        # Metrics Display
        st.markdown("---")
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        with detail_col1: st.metric("Action Taken", current_day_data["Action"])
        with detail_col2: st.metric("Target Tumor Size", f"{target_size}")
        with detail_col3: st.metric("Drug A Resistance", f"{current_day_data['Resist_A']:.2f}")
        with detail_col4: st.metric("Drug B Resistance", f"{current_day_data['Resist_B']:.2f}")
        
        # Strategy Graph
        st.subheader("📈 Evolutionary Trap Analysis")
        df_history = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0E1117')
        ax1.set_facecolor('#161B22')
        
        ax1.plot(df_history['Day'], df_history['Tumor Size'], color='tab:red', linewidth=3, label="Tumor Size", marker='o')
        ax1.set_ylabel('Tumor Size', color='tab:red')
        
        ax2 = ax1.twinx()
        ax2.plot(df_history['Day'], df_history['Resist_A'], '--', label="Resist A", color='#1f77b4')
        ax2.plot(df_history['Day'], df_history['Resist_B'], ':', label="Resist B", color='#2ca02c')
        ax2.axhline(y=2.5, color='#FFA500', linestyle='-', label="Trap Threshold (2.5)")
        ax2.set_ylabel('Resistance Levels', color='#C9D1D9')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#161B22', edgecolor='#30363D')
        
        st.pyplot(fig)
else:
    st.warning("Please upload a CSV file to begin analysis.")