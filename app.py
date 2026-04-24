import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import PatientAnalyzer
from environment import CancerSimulation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# --- HELPER FUNCTION: TUMOR VISUALIZATION ---
def create_tumor_visualization(tumor_size, res_level_or_list, max_res=15.0):
    """
    High-performance tumor cell visualization using Plotly Scattergl.
    """
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
                    size=6,
                    color=norm_resistances,
                    colorscale='YlOrRd',
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title='Resistance', thickness=12, len=0.65, y=0.5, tickfont=dict(color='#C9D1D9')),
                    opacity=0.85,
                    line=dict(width=0)
                ),
                hovertemplate='Resistance: %{marker.color:.2f}<extra></extra>'
            )
        ]
    )

    fig.update_layout(
        template=None,
        paper_bgcolor='#0E1117',
        plot_bgcolor='#161B22',
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor='x'),
        title=dict(text=f'Tumor Cells: {num_cells:,} | Avg Resistance: {avg_resistance:.1f}', x=0.5, font=dict(color='#C9D1D9', size=14)),
        font=dict(color='#C9D1D9'),
    )

    return fig

# --- CONFIGURATION ---
st.set_page_config(
    page_title="OncoSteer: Evolutionary AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "peacekeeper_final_azure" 
DEFAULT_DATA_PATH = "data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"

@st.cache_resource
def load_model():
    if os.path.exists(f"{MODEL_PATH}.zip"):
        return PPO.load(MODEL_PATH)
    else:
        st.error(f"Model file {MODEL_PATH}.zip not found!")
        return None

st.title("🛡️ OncoSteer")
st.markdown("### Steering tumor evolution toward therapeutic vulnerability")
st.info("Upload patient proteomic data or use the default synthetic dataset to generate a personalized treatment plan.")

# --- SIDEBAR: DATA SELECTION ---
st.sidebar.header("Patient Data Input")
data_mode = st.sidebar.radio("Select Data Source:", ["Upload CSV", "Use Default Synthetic Data"])

uploaded_file = None
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Proteomic CSV", type=["csv"])
    current_data_source = uploaded_file.name if uploaded_file else None
else:
    if os.path.exists(DEFAULT_DATA_PATH):
        uploaded_file = DEFAULT_DATA_PATH
        current_data_source = "default_synthetic_data"
    else:
        st.sidebar.error("Default data file not found at path.")

# Initialize session state
if 'treatment_history' not in st.session_state:
    st.session_state.treatment_history = None
if 'current_day_view' not in st.session_state:
    st.session_state.current_day_view = 0
if 'last_loaded_file' not in st.session_state:
    st.session_state.last_loaded_file = None

# Process Data
if uploaded_file is not None:
    # Reset if source changed
    if current_data_source != st.session_state.last_loaded_file:
        st.session_state.treatment_history = None
        st.session_state.current_day_view = 0
        st.session_state.last_loaded_file = current_data_source

    data = pd.read_csv(uploaded_file)
    model = load_model()
    
    try:
        analyzer = PatientAnalyzer(df=data)
        with st.spinner("Analyzing Proteomic Signatures..."):
            profile = analyzer.get_patient_profile(data)
            cell_resistance_data = analyzer.get_cell_resistance_data()
            st.session_state.patient_profile = profile
            st.session_state.cell_resistance_data = cell_resistance_data
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        st.stop()
    
    # Display Diagnostic Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Resistance (Drug A)", f"{profile['max_res_a']:.2f}")
    col2.metric("Avg Growth Rate", f"{profile['avg_growth']:.2f}%")
    col3.metric("Initial Res_B", "5.00 (Standard)")

    if st.button("Generate Optimized Treatment Plan"):
        env = CancerSimulation(profile)
        obs, _ = env.reset()
        history = []
        day = 1
        
        while day <= 30:
            # Current size before treatment
            size_before = int(obs[0])
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Size after treatment
            size_after = int(obs[0])
            
            action_names = {0: "Rest (Recovery)", 1: "Drug A (Priming)", 2: "Drug B (TRAP)"}
            
            history.append({
                "Day": day,
                "Action": action_names[action],
                "Size Before": size_before,
                "Size After": size_after,
                "Tumor Size": size_after, # For backward compatibility
                "Resist_A": round(obs[1], 2),
                "Resist_B": round(obs[2], 2),
            })
            
            if size_after <= 0 or terminated: break
            day += 1
        
        st.session_state.treatment_history = history

    # --- RESULTS DISPLAY ---
    if st.session_state.treatment_history:
        history = st.session_state.treatment_history
        
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        
        # Day Navigation
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("◀ PREV") and st.session_state.current_day_view > 0:
                st.session_state.current_day_view -= 1
                st.rerun()
        with nav_col2:
            st.markdown(f"<h3 style='text-align: center;'>📅 Day {st.session_state.current_day_view + 1}</h3>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("NEXT ▶") and st.session_state.current_day_view < len(history) - 1:
                st.session_state.current_day_view += 1
                st.rerun()

        curr = history[st.session_state.current_day_view]
        
        # Side-by-Side Visualization
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_tumor_visualization(curr["Size Before"], st.session_state.cell_resistance_data), use_container_width=True, key="before")
        with vis_col2:
            st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_tumor_visualization(curr["Size After"], st.session_state.cell_resistance_data), use_container_width=True, key="after")

        # --- UPDATED CHART: BEFORE VS AFTER ---
        st.markdown("---")
        st.subheader("📈 Evolutionary Impact Analysis")
        
        df_hist = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0E1117')
        ax.set_facecolor('#161B22')

        x = np.arange(len(df_hist))
        width = 0.35

        ax.bar(x - width/2, df_hist['Size Before'], width, label='Size Before Drug', color='#E74C3C', alpha=0.7)
        ax.bar(x + width/2, df_hist['Size After'], width, label='Size After Drug', color='#2ECC71', alpha=0.9)

        ax.set_xlabel('Treatment Day', color='#C9D1D9')
        ax.set_ylabel('Tumor Cell Count', color='#C9D1D9')
        ax.set_title('Tumor Size Reduction per Day: Before vs. After Dose', color='#C9D1D9', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df_hist['Day'], color='#C9D1D9')
        ax.tick_params(axis='y', labelcolor='#C9D1D9')
        ax.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')
        ax.grid(axis='y', alpha=0.1)

        plt.tight_layout()
        st.pyplot(fig)

        # Full History Table
        st.subheader("📊 Full Treatment Timeline")
        st.dataframe(df_hist, use_container_width=True)

else:
    st.warning("Please upload a CSV file or select the default dataset to begin analysis.")