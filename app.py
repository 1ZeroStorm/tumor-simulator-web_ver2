import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import PatientAnalyzer
from environment import CancerSimulation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import io

# --- HELPER FUNCTION: TUMOR VISUALIZATION ---
def create_tumor_visualization(tumor_size, res_level_or_list, max_res=15.0, jelly_intensity=0.0, jelly_phase=0):
    """
    High-performance tumor cell visualization.
    Each dot = 1 tumor cell. Color gradient based on resistance (Blue -> Red).
    
    res_level_or_list: Either a single float (average resistance) or list of individual resistances
    """
    # 1. Initialize persistent cell coordinate pool in session state
    # This ensures coordinates don't change when switching days (the "jelly" effect)
    if 'cell_coordinates' not in st.session_state:
        # Create a large pool of random cell positions (up to 20,000 cells)
        st.session_state.cell_coordinates = np.random.rand(20000, 2)
    
    # 2. Get exact number of cells to display
    num_cells = int(min(len(st.session_state.cell_coordinates), max(1, tumor_size)))
    
    # Use the first N cells from our persistent pool
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()

    # Apply a jelly-like motion effect when requested
    if jelly_intensity > 0 and num_cells > 0:
        intensity = min(0.22, jelly_intensity)
        phase = jelly_phase * np.pi / 4
        offsets_x = np.sin(2 * np.pi * (cell_coords[:, 0] + phase)) * intensity * (1 - cell_coords[:, 1])
        offsets_y = np.cos(2 * np.pi * (cell_coords[:, 1] + phase)) * intensity * (1 - cell_coords[:, 0])
        cell_coords = np.clip(cell_coords + np.stack([offsets_x, offsets_y], axis=1), 0, 1)
    
    # 3. Determine resistance values for each cell
    if isinstance(res_level_or_list, (list, np.ndarray)):
        # Use individual cell resistance levels
        cell_resistances = np.array(res_level_or_list[:num_cells])
        # Pad with average if we have fewer resistance values than cells
        if len(cell_resistances) < num_cells:
            avg_res = np.mean(cell_resistances) if len(cell_resistances) > 0 else max_res / 2
            cell_resistances = np.pad(cell_resistances, (0, num_cells - len(cell_resistances)), 
                                     constant_values=avg_res)
        avg_resistance = np.mean(cell_resistances)
    else:
        # Use single resistance level for all cells
        cell_resistances = np.full(num_cells, res_level_or_list)
        avg_resistance = res_level_or_list
    
    # 4. Create color gradient based on individual cell resistance levels
    norm_resistances = np.clip(cell_resistances / max_res, 0, 1)
    
    # 5. Render the tumor visualization
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0E1117', dpi=80)
    ax.set_facecolor('#161B22')
    
    # Plot each cell as a small dot with the gradient colors
    # Pass the normalized resistance values and let matplotlib apply the colormap
    scatter = ax.scatter(
        cell_coords[:, 0],
        cell_coords[:, 1],
        s=12,  # Slightly larger dots for better color visibility
        c=norm_resistances,  # Pass normalized values directly
        cmap='YlOrRd',  # Yellow-Orange-Red gradient
        alpha=0.95,  # Higher opacity for better color contrast
        edgecolors='none',
        vmin=0,
        vmax=1
    )
    
    # Style the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add info text
    title_text = f"Tumor Cells: {num_cells:,} | Average Resistance: {avg_resistance:.1f}"
    fig.text(0.5, 0.95, title_text, ha='center', fontsize=12, 
             color='#C9D1D9', weight='bold')
    
    # Add resistance status indicator
    norm_avg = min(1.0, avg_resistance / max_res)
    if norm_avg > 0.7:
        status = "⚠️ HIGH RESISTANCE"
        color = '#FF2222'
    else:
        status = "✓ LOW RESISTANCE"
        color = '#22FF22'
    
    fig.text(0.5, 0.02, status, ha='center', fontsize=11, color=color, weight='bold')
    
    return fig


def init_tumor_canvas(tumor_size, res_level_or_list, max_res=15.0, jelly_intensity=0.0, jelly_phase=0):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0E1117', dpi=80)
    ax.set_facecolor('#161B22')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    if 'cell_coordinates' not in st.session_state:
        st.session_state.cell_coordinates = np.random.rand(20000, 2)

    num_cells = int(min(len(st.session_state.cell_coordinates), max(1, tumor_size)))
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()
    if jelly_intensity > 0 and num_cells > 0:
        intensity = min(0.22, jelly_intensity)
        phase = jelly_phase * np.pi / 4
        offsets_x = np.sin(2 * np.pi * (cell_coords[:, 0] + phase)) * intensity * (1 - cell_coords[:, 1])
        offsets_y = np.cos(2 * np.pi * (cell_coords[:, 1] + phase)) * intensity * (1 - cell_coords[:, 0])
        cell_coords = np.clip(cell_coords + np.stack([offsets_x, offsets_y], axis=1), 0, 1)

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
    scatter = ax.scatter(
        cell_coords[:, 0],
        cell_coords[:, 1],
        s=12,
        c=norm_resistances,
        cmap='YlOrRd',
        alpha=0.95,
        edgecolors='none',
        vmin=0,
        vmax=1,
    )

    title_text = fig.text(0.5, 0.95, f"Tumor Cells: {num_cells:,} | Average Resistance: {avg_resistance:.1f}",
                          ha='center', fontsize=12, color='#C9D1D9', weight='bold')
    norm_avg = min(1.0, avg_resistance / max_res)
    if norm_avg > 0.7:
        status = "⚠️ HIGH RESISTANCE"
        status_color = '#FF2222'
    else:
        status = "✓ LOW RESISTANCE"
        status_color = '#22FF22'
    status_text = fig.text(0.5, 0.02, status, ha='center', fontsize=11, color=status_color, weight='bold')

    return fig, ax, scatter, title_text, status_text


def update_tumor_canvas(fig, scatter, title_text, status_text, tumor_size, res_level_or_list, max_res=15.0, jelly_intensity=0.0, jelly_phase=0):
    if 'cell_coordinates' not in st.session_state:
        st.session_state.cell_coordinates = np.random.rand(20000, 2)

    num_cells = int(min(len(st.session_state.cell_coordinates), max(1, tumor_size)))
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()
    if jelly_intensity > 0 and num_cells > 0:
        intensity = min(0.22, jelly_intensity)
        phase = jelly_phase * np.pi / 4
        offsets_x = np.sin(2 * np.pi * (cell_coords[:, 0] + phase)) * intensity * (1 - cell_coords[:, 1])
        offsets_y = np.cos(2 * np.pi * (cell_coords[:, 1] + phase)) * intensity * (1 - cell_coords[:, 0])
        cell_coords = np.clip(cell_coords + np.stack([offsets_x, offsets_y], axis=1), 0, 1)

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
    scatter.set_offsets(cell_coords)
    scatter.set_array(norm_resistances)

    title_text.set_text(f"Tumor Cells: {num_cells:,} | Average Resistance: {avg_resistance:.1f}")
    norm_avg = min(1.0, avg_resistance / max_res)
    if norm_avg > 0.7:
        status = "⚠️ HIGH RESISTANCE"
        status_color = '#FF2222'
    else:
        status = "✓ LOW RESISTANCE"
        status_color = '#22FF22'
    status_text.set_text(status)
    status_text.set_color(status_color)

    fig.canvas.draw_idle()
    return fig


def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=80)
    buf.seek(0)
    return buf

# --- CONFIGURATION ---
st.set_page_config(
    page_title="The Peacekeeper: AI Immunotherapy",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "peacekeeper_final_azure" # Do not add .zip here

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

# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = None
if 'treatment_history' not in st.session_state:
    st.session_state.treatment_history = None
if 'current_day_view' not in st.session_state:
    st.session_state.current_day_view = 0
if 'previous_day_view' not in st.session_state:
    st.session_state.previous_day_view = 0
if 'animate_transition' not in st.session_state:
    st.session_state.animate_transition = False
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'cell_resistance_data' not in st.session_state:
    st.session_state.cell_resistance_data = None

if uploaded_file is not None:
    # Check if a new file was uploaded (different from the current one)
    if uploaded_file.name != st.session_state.current_file_name:
        # Reset treatment results when a new file is uploaded
        st.session_state.treatment_history = None
        st.session_state.current_day_view = 0
        st.session_state.current_file_name = uploaded_file.name
    
    # Store uploaded data in session state
    data = pd.read_csv(uploaded_file)
    st.session_state.uploaded_data = data
    
    model = load_model()
    
    try:
        analyzer = PatientAnalyzer(df=data)
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        st.stop()
    
    # Step 1: Diagnostic Phase (Neural Network)
    with st.spinner("Analyzing Proteomic Signatures..."):
        profile = analyzer.get_patient_profile(data)
        cell_resistance = analyzer.get_cell_resistance_data()
        st.session_state.patient_profile = profile
        st.session_state.cell_resistance_data = cell_resistance
    
    # --- DISPLAY DIAGNOSTIC SUMMARY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Resistance (Drug A)", f"{profile['max_res_a']:.2f}")
    col2.metric("Avg Growth Rate", f"{profile['avg_growth']:.2f}%")
    col3.metric("Initial Res_B", "5.00 (Standard)")

    if st.button("Generate Optimized Treatment Plan"):
        # Step 2: Strategy Optimization Phase
        env = CancerSimulation(profile)
        obs, _ = env.reset()
        
        history = []
        terminated = False
        day = 1
        
        # Run simulation for up to 30 days
        while not terminated and day <= 30:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Map action to readable name - clear conditional logic
            if action == 0:
                action_name = "Rest (Recovery)"
            elif action == 1:
                action_name = "Drug A (Priming)"
            else:  # action == 2
                action_name = "Drug B (TRAP)"
            
            # Get values from state
            size, res_a, res_b = obs[0], obs[1], obs[2]
            toxicity = info.get('toxicity', day)
            
            # Safety Status Logic
            if toxicity < 5: status = "🟢 SAFE"
            elif toxicity < 8: status = "🟡 MONITOR"
            else: status = "🔴 CRITICAL"
            
            history.append({
                "Day": day,
                "Action": action_name,
                "Tumor Size": int(size),
                "Resist_A": round(res_a, 2),
                "Resist_B": round(res_b, 2),
                "Safety Status": status
            })
            
            if size <= 0:
                break
            day += 1
        
        # Store history in session state
        st.session_state.treatment_history = history
        st.session_state.current_day_view = 0
    
    # Display results if history exists
    if st.session_state.treatment_history is not None:
        history = st.session_state.treatment_history
        profile = st.session_state.patient_profile
        
        # --- RESULTS DISPLAY ---
        st.success(f"Strategy Optimized: Treatment targets eradication by Day {len(history)}")
        
        # === INTERACTIVE TUMOR VISUALIZATION ===
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        
        # Color Legend
        st.markdown("""
        <div style='padding: 16px; border-radius: 14px; background-color: #0D1117; color: #C9D1D9; border: 1px solid #30363D;'>
          <strong>Resistance Level Gradient:</strong>
          <div style='margin: 12px 0; height: 20px; border-radius: 12px; background: linear-gradient(90deg, #FFFFB2 0%, #FED976 25%, #FD8D3C 50%, #F03B20 75%, #BD0026 100%);'></div>
          <div style='display: flex; justify-content: space-between; font-size: 0.95rem;'>
            <span>Low resistance</span>
            <span>High resistance</span>
          </div>
          <div style='margin-top: 10px; font-size: 0.95rem;'>
            Each dot's color is mapped to resistance level on a continuous scale, from cooler low-resistance tones to warmer high-resistance tones.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Day Navigation Controls
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])
        
        with nav_col1:
            if st.button("◀ PREV", key="prev_day"):
                if st.session_state.current_day_view > 0:
                    st.session_state.previous_day_view = st.session_state.current_day_view
                    st.session_state.current_day_view -= 1
                    st.session_state.animate_transition = True
                    st.rerun()
        
        with nav_col3:
            st.markdown(f"<div style='text-align: center; padding: 10px; margin-top: 8px;'><h3>📅 Day {st.session_state.current_day_view + 1}</h3></div>", 
                       unsafe_allow_html=True)
        
        with nav_col5:
            if st.button("NEXT ▶", key="next_day"):
                if st.session_state.current_day_view < len(history) - 1:
                    st.session_state.previous_day_view = st.session_state.current_day_view
                    st.session_state.current_day_view += 1
                    st.session_state.animate_transition = True
                    st.rerun()
        
        # Get current day data
        current_idx = st.session_state.current_day_view
        current_day_data = history[current_idx]
        res_level = current_day_data["Resist_A"]
        tumor_size = current_day_data["Tumor Size"]
        cell_resistance = st.session_state.cell_resistance_data or res_level
        
        # Tumor animation: sliced jelly motion on navigation and growth
        st.markdown("---")
        animation_placeholder = st.empty()
        with animation_placeholder.container():
            animate_transition = st.session_state.animate_transition
            if animate_transition:
                prev_size = history[st.session_state.previous_day_view]["Tumor Size"]
                fig, ax, scatter, title_text, status_text = init_tumor_canvas(prev_size, cell_resistance)
                steps = 8
                size_diff = tumor_size - prev_size
                growth_factor = abs(size_diff) / max(prev_size, 1)
                for frame in range(steps + 1):
                    t = frame / steps
                    interpolated_size = int(prev_size + size_diff * t)
                    jelly_intensity = 0.06 + min(0.35, growth_factor * 0.35)
                    if size_diff > 0:
                        jelly_intensity += 0.06
                    update_tumor_canvas(
                        fig,
                        scatter,
                        title_text,
                        status_text,
                        interpolated_size,
                        cell_resistance,
                        jelly_intensity=jelly_intensity,
                        jelly_phase=frame,
                    )
                    animation_placeholder.image(fig_to_image_bytes(fig), use_column_width=True)
                    if frame < steps:
                        time.sleep(0.07)
                st.session_state.animate_transition = False
                plt.close(fig)
            else:
                fig, ax, scatter, title_text, status_text = init_tumor_canvas(tumor_size, cell_resistance)
                animation_placeholder.image(fig_to_image_bytes(fig), use_column_width=True)
                plt.close(fig)
        
        st.markdown("---")
        toggle_col1, toggle_col2 = st.columns([1, 1])
        with toggle_col1:
            show_before = st.toggle("🔴 Before Drug Application", value=True, key="before_toggle")
        with toggle_col2:
            show_after = st.toggle("🟢 After Drug Application", value=True, key="after_toggle")
        
        # Display tumor visualizations
        if show_before and show_after:
            vis_col1, vis_col2 = st.columns(2)
            
            # Estimate tumor size before drug (slightly larger)
            tumor_before = int(current_day_data["Tumor Size"] * 1.15)
            
            with vis_col1:
                st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
                fig_before = create_tumor_visualization(tumor_before, cell_resistance)
                st.pyplot(fig_before)
                plt.close(fig_before)
            
            with vis_col2:
                st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
                fig_after = create_tumor_visualization(current_day_data["Tumor Size"], cell_resistance)
                st.pyplot(fig_after)
                plt.close(fig_after)
        
        elif show_before:
            st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
            tumor_before = int(current_day_data["Tumor Size"] * 1.15)
            fig_before = create_tumor_visualization(tumor_before, cell_resistance)
            st.pyplot(fig_before)
            plt.close(fig_before)
        
        elif show_after:
            st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
            fig_after = create_tumor_visualization(current_day_data["Tumor Size"], cell_resistance)
            st.pyplot(fig_after)
            plt.close(fig_after)
        
        # Display current day details
        st.markdown("---")
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        
        with detail_col1:
            st.metric("Action Taken", current_day_data["Action"], 
                     delta=None, delta_color="off")
        with detail_col2:
            st.metric("Tumor Size", f"{current_day_data['Tumor Size']}", 
                     delta=None, delta_color="off")
        with detail_col3:
            st.metric("Drug A Resistance", f"{current_day_data['Resist_A']:.2f}", 
                     delta=None, delta_color="off")
        with detail_col4:
            st.metric("Drug B Resistance", f"{current_day_data['Resist_B']:.2f}", 
                     delta=None, delta_color="off")
        
        # Display Table
        st.markdown("---")
        st.subheader("📊 Full Treatment Timeline")
        df_history = pd.DataFrame(history)
        st.dataframe(df_history, use_container_width=True)
        
        # --- VISUALIZATION (The "Why") ---
        st.subheader("📈 Evolutionary Trap Analysis - Treatment Strategy Over Time")
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0E1117')
        ax1.set_facecolor('#161B22')

        # Identify Drug B application days
        drug_b_days = df_history[df_history['Action'].str.contains('Drug B', na=False)]['Day'].tolist()
        
        # Plot Tumor Size on primary y-axis
        color = 'tab:red'
        ax1.set_xlabel('Day', color='#C9D1D9', fontsize=11)
        ax1.set_ylabel('Tumor Size', color=color, fontsize=11)
        ax1.plot(df_history['Day'], df_history['Tumor Size'], color=color, linewidth=3, label="Tumor Size", marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', labelcolor='#C9D1D9')
        ax1.grid(True, alpha=0.2, color='#30363D')
        
        # Highlight Drug B application areas
        if drug_b_days:
            for drug_b_day in drug_b_days:
                ax1.axvline(x=drug_b_day, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.6)
            ax1.axvspan(min(drug_b_days) - 0.5, max(drug_b_days) + 0.5, alpha=0.1, color='#FF6B6B', label='Drug B (TRAP) Applied')

        # Plot Resistance levels on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance Levels', color='#C9D1D9', fontsize=11)
        ax2.plot(df_history['Day'], df_history['Resist_A'], '--', label="Resist A (Priming Target)", 
                color='#1f77b4', linewidth=2, marker='s', markersize=3)
        ax2.plot(df_history['Day'], df_history['Resist_B'], ':', label="Resist B (Trap Trigger)", 
                color='#2ca02c', linewidth=2.5, marker='^', markersize=4)
        
        # Add trap threshold line
        ax2.axhline(y=2.5, color='#FFA500', linestyle='-', linewidth=2, alpha=0.7, label="Trap Threshold (2.5)")
        ax2.tick_params(axis='y', labelcolor='#C9D1D9')
        ax2.set_ylim(bottom=0)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, facecolor='#161B22', edgecolor='#30363D')
        
        ax1.set_title('Strategic Treatment Timeline: Priming → Trap Activation', 
                     color='#C9D1D9', fontsize=12, weight='bold', pad=15)
        
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **📋 Treatment Strategy Explanation:**
        
        🔵 **Phase 1 - Priming (Drug A):** The AI applies Drug A to build up collateral sensitivity. 
        While Res_A increases, Res_B is driven down toward the trap threshold.
        
        🔴 **Phase 2 - Trap Activation (Drug B):** Once Res_B crosses the 2.5 threshold, 
        the evolutionary trap is sprung! Drug B becomes highly effective (85% kill rate), 
        rapidly reducing tumor burden while maintaining safety.
        
        ⚠️ **Toxicity Management:** The system monitors cumulative toxicity and switches 
        to rest days when needed to prevent organ damage.
        """)

else:
    st.warning("Please upload a CSV file to begin analysis.")