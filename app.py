import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import PatientAnalyzer
from environment import CancerSimulation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os

# --- HELPER FUNCTION: TUMOR VISUALIZATION (STATIC PLOTLY) ---
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

# --- HELPER FUNCTION: HTML/JS PHYSICS ANIMATION ---
def physics_tumor_visualization(prev_size, target_size, resistance_level, max_res=15.0):
    """
    Renders an HTML5 Canvas that animates duplication with non-uniform (decelerating) physics.
    """
    # Normalize resistance for coloring (0 to 1)
    norm_res = min(max(resistance_level / max_res, 0), 1)
    
    # Calculate a color based on resistance (Yellow to Dark Red)
    r = int(255)
    g = int(255 - (255 * norm_res))
    b = int(178 - (178 * norm_res))
    hex_color = f"#{r:02x}{g:02x}{b:02x}"

    html_code = f"""
    <div style="background-color: #161B22; border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #30363D;">
        <p style="color: #C9D1D9; font-family: sans-serif; font-weight: bold; margin-bottom: 5px;">
            Dynamic Evolution | Target Cells: {target_size} | Resistance: {resistance_level:.2f}
        </p>
        <canvas id="tumorCanvas" width="700" height="400"></canvas>
    </div>
    
    <script>
        const canvas = document.getElementById('tumorCanvas');
        const ctx = canvas.getContext('2d');
        
        let particles = [];
        const prevSize = {prev_size};
        const targetSize = {target_size};
        const color = "{hex_color}";
        
        // Initialize with previous size
        for(let i = 0; i < prevSize; i++) {{
            particles.push({{
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: 0,
                vy: 0
            }});
        }}
        
        // Handle growth (duplication with physics)
        if (targetSize > prevSize) {{
            const diff = targetSize - prevSize;
            for(let i = 0; i < diff; i++) {{
                // Pick a random existing parent to "duplicate" from
                let parent = particles[Math.floor(Math.random() * prevSize)] || {{x: canvas.width/2, y: canvas.height/2}};
                
                // Explosive duplication force
                let angle = Math.random() * Math.PI * 2;
                let force = Math.random() * 8 + 4; // Initial burst speed
                
                particles.push({{
                    x: parent.x,
                    y: parent.y,
                    vx: Math.cos(angle) * force,
                    vy: Math.sin(angle) * force
                }});
            }}
        }} 
        // Handle shrinking (cells die off)
        else if (targetSize < prevSize) {{
            particles = particles.slice(0, targetSize);
        }}

        function animate() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            for(let i = 0; i < particles.length; i++) {{
                let p = particles[i];
                
                // Physics: Apply Non-uniform motion (Friction/Deceleration)
                p.x += p.vx;
                p.y += p.vy;
                p.vx *= 0.90; // Deceleration factor
                p.vy *= 0.90;
                
                // Bounds checking
                if(p.x < 0) p.x = canvas.width;
                if(p.x > canvas.width) p.x = 0;
                if(p.y < 0) p.y = canvas.height;
                if(p.y > canvas.height) p.y = 0;
                
                // Draw Cell
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.85;
                ctx.fill();
            }}
            
            requestAnimationFrame(animate);
        }}
        
        animate();
    </script>
    """
    components.html(html_code, height=470)


# --- CONFIGURATION ---
st.set_page_config(page_title="The Peacekeeper: AI Immunotherapy", layout="wide", initial_sidebar_state="expanded")
MODEL_PATH = "peacekeeper_final_azure"

@st.cache_resource
def load_model():
    if os.path.exists(f"{MODEL_PATH}.zip"):
        return PPO.load(MODEL_PATH)
    else:
        st.error(f"Model file {MODEL_PATH}.zip not found in repository!")
        return None

st.title("🛡️ The Peacekeeper")
st.markdown("### Digital Immunotherapy & Evolutionary Trap Optimizer")
st.info("Upload patient proteomic data to generate a personalized, safety-constrained treatment plan.")

# --- SIDEBAR: PATIENT DATA ---
st.sidebar.header("Patient Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Proteomic CSV", type=["csv"])

if 'uploaded_data' not in st.session_state: st.session_state.uploaded_data = None
if 'patient_profile' not in st.session_state: st.session_state.patient_profile = None
if 'treatment_history' not in st.session_state: st.session_state.treatment_history = None
# Replaced 'current_day_view' with 'current_phase_step' to track before/after states
if 'current_phase_step' not in st.session_state: st.session_state.current_phase_step = 0
if 'current_file_name' not in st.session_state: st.session_state.current_file_name = None
if 'cell_resistance_data' not in st.session_state: st.session_state.cell_resistance_data = None

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.current_file_name:
        st.session_state.treatment_history = None
        st.session_state.current_phase_step = 0
        st.session_state.current_file_name = uploaded_file.name
    
    data = pd.read_csv(uploaded_file)
    st.session_state.uploaded_data = data
    model = load_model()
    
    try:
        analyzer = PatientAnalyzer(df=data)
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        st.stop()
    
    with st.spinner("Analyzing Proteomic Signatures..."):
        profile = analyzer.get_patient_profile(data)
        cell_resistance = analyzer.get_cell_resistance_data()
        st.session_state.patient_profile = profile
        st.session_state.cell_resistance_data = cell_resistance
    
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
            toxicity = info.get('toxicity', day)
            status = "🟢 SAFE" if toxicity < 5 else "🟡 MONITOR" if toxicity < 8 else "🔴 CRITICAL"
            
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
    
    if st.session_state.treatment_history is not None:
        history = st.session_state.treatment_history
        max_steps = len(history) * 2 - 1 # Each day has a 'before' (even) and 'after' (odd) state
        
        st.success(f"Strategy Optimized: Treatment targets eradication by Day {len(history)}")
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        
        # Day Navigation Controls (Step-based: Before -> After -> Before -> After...)
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])
        
        with nav_col1:
            if st.button("◀ PREV", key="prev_day"):
                if st.session_state.current_phase_step > 0:
                    st.session_state.current_phase_step -= 1
                    st.rerun()
        
        # Calculate current day and whether we are in the "Before" or "After" phase
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
        
        # Logic for animation sizing
        base_size = current_day_data["Tumor Size"]
        res_level = current_day_data["Resist_A"]
        
        # Calculate the previous size for the animation delta
        if st.session_state.current_phase_step == 0:
            prev_size = int(base_size * 1.15) # Genesis state
            target_size = prev_size
        else:
            if not is_after_treatment:
                # We are looking at "Before" of Day N. The previous state was "After" of Day N-1
                prev_size = history[current_idx - 1]["Tumor Size"]
                target_size = int(base_size * 1.15) # Tumor grew a bit overnight
            else:
                # We are looking at "After" of Day N. The previous state was "Before" of Day N
                prev_size = int(base_size * 1.15)
                target_size = base_size

        # MAIN DISPLAY: Physics-based Canvas Animation
        st.markdown("---")
        physics_tumor_visualization(prev_size, target_size, res_level)
        
        # STATIC BEFORE/AFTER PANELS (Kept intact per your request)
        st.markdown("---")
        toggle_col1, toggle_col2 = st.columns([1, 1])
        with toggle_col1:
            show_before = st.toggle("🔴 Show Static Before", value=False, key="before_toggle")
        with toggle_col2:
            show_after = st.toggle("🟢 Show Static After", value=False, key="after_toggle")
        
        cell_resistance = st.session_state.cell_resistance_data or res_level
        
        if show_before and show_after:
            vis_col1, vis_col2 = st.columns(2)
            tumor_before = int(current_day_data["Tumor Size"] * 1.15)
            with vis_col1:
                st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
                fig_before = create_tumor_visualization(tumor_before, cell_resistance)
                st.plotly_chart(fig_before, use_container_width=True, key="tumor_plot_before")
            with vis_col2:
                st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
                fig_after = create_tumor_visualization(current_day_data["Tumor Size"], cell_resistance)
                st.plotly_chart(fig_after, use_container_width=True, key="tumor_plot_after")
        elif show_before:
            st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
            tumor_before = int(current_day_data["Tumor Size"] * 1.15)
            fig_before = create_tumor_visualization(tumor_before, cell_resistance)
            st.plotly_chart(fig_before, use_container_width=True, key="tumor_plot_before")
        elif show_after:
            st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
            fig_after = create_tumor_visualization(current_day_data["Tumor Size"], cell_resistance)
            st.plotly_chart(fig_after, use_container_width=True, key="tumor_plot_after")
        
        # Metrics Display
        st.markdown("---")
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        with detail_col1: st.metric("Action Taken", current_day_data["Action"])
        with detail_col2: st.metric("Target Tumor Size", f"{target_size}")
        with detail_col3: st.metric("Drug A Resistance", f"{current_day_data['Resist_A']:.2f}")
        with detail_col4: st.metric("Drug B Resistance", f"{current_day_data['Resist_B']:.2f}")
        
        # History Table
        st.markdown("---")
        st.subheader("📊 Full Treatment Timeline")
        df_history = pd.DataFrame(history)
        st.dataframe(df_history, use_container_width=True)
        
        # Strategy Graph
        st.subheader("📈 Evolutionary Trap Analysis")
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0E1117')
        ax1.set_facecolor('#161B22')
        drug_b_days = df_history[df_history['Action'].str.contains('Drug B', na=False)]['Day'].tolist()
        
        color = 'tab:red'
        ax1.set_xlabel('Day', color='#C9D1D9', fontsize=11)
        ax1.set_ylabel('Tumor Size', color=color, fontsize=11)
        ax1.plot(df_history['Day'], df_history['Tumor Size'], color=color, linewidth=3, label="Tumor Size", marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', labelcolor='#C9D1D9')
        ax1.grid(True, alpha=0.2, color='#30363D')
        
        if drug_b_days:
            for drug_b_day in drug_b_days:
                ax1.axvline(x=drug_b_day, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.6)
            ax1.axvspan(min(drug_b_days) - 0.5, max(drug_b_days) + 0.5, alpha=0.1, color='#FF6B6B', label='Drug B Applied')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance Levels', color='#C9D1D9', fontsize=11)
        ax2.plot(df_history['Day'], df_history['Resist_A'], '--', label="Resist A", color='#1f77b4', linewidth=2)
        ax2.plot(df_history['Day'], df_history['Resist_B'], ':', label="Resist B", color='#2ca02c', linewidth=2.5)
        ax2.axhline(y=2.5, color='#FFA500', linestyle='-', linewidth=2, alpha=0.7, label="Threshold (2.5)")
        ax2.tick_params(axis='y', labelcolor='#C9D1D9')
        ax2.set_ylim(bottom=0)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, facecolor='#161B22', edgecolor='#30363D')
        
        fig.tight_layout()
        st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to begin analysis.")