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
def create_tumor_visualization(tumor_size, resistance_list, max_res=15.0):
    """
    High-performance tumor cell visualization.
    Maps individual resistance values to colors for every single cell.
    """
    if 'cell_coordinates' not in st.session_state:
        # Persistent coordinate pool to keep cell positions stable during refresh
        st.session_state.cell_coordinates = np.random.rand(20000, 2)

    num_cells = int(min(len(st.session_state.cell_coordinates), max(1, tumor_size)))
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()

    # 1. Handle Resistance Data
    # If we have a list of individual resistances, we slice it to match current population
    if isinstance(resistance_list, (list, np.ndarray)) and len(resistance_list) > 0:
        # We cycle or pad the list if the current tumor size exceeds our data sample
        if len(resistance_list) < num_cells:
            repeats = (num_cells // len(resistance_list)) + 1
            current_resistances = np.tile(resistance_list, repeats)[:num_cells]
        else:
            current_resistances = np.array(resistance_list[:num_cells])
    else:
        # Fallback if no list is provided
        current_resistances = np.full(num_cells, 5.0)

    # 2. Normalize for colorscale (0 to 1)
    norm_colors = np.clip(current_resistances / max_res, 0, 1)

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=cell_coords[:, 0],
                y=cell_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=norm_colors,
                    colorscale='YlOrRd', 
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(
                        title='Individual Resistance', 
                        thickness=10, 
                        tickfont=dict(color='#888888')
                    ),
                    opacity=0.8
                ),
                hovertemplate='Cell Resistance: %{marker.color:.2f}<extra></extra>'
            )
        ]
    )

    fig.update_layout(
        template=None,
        paper_bgcolor='#0E1117',
        plot_bgcolor='#161B22',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        title=dict(
            text=f"Live Population: {num_cells:,} cells", 
            x=0.5, 
            font=dict(color='#888888', size=14)
        ),
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
    return None

st.title("🛡️ OncoSteer")
st.markdown("### Steering tumor evolution toward therapeutic vulnerability")

# --- SIDEBAR ---
st.sidebar.header("Patient Data Input")
data_mode = st.sidebar.radio("Select Data Source:", ["Upload CSV", "Use Default Synthetic Data"])

uploaded_file = None
current_data_source = None

if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Proteomic CSV", type=["csv"])
    if uploaded_file:
        current_data_source = uploaded_file.name
else:
    if os.path.exists(DEFAULT_DATA_PATH):
        uploaded_file = DEFAULT_DATA_PATH
        current_data_source = "default_synthetic_data"
    else:
        st.sidebar.error("Default data file not found.")

if 'treatment_history' not in st.session_state:
    st.session_state.treatment_history = None
if 'cell_res_data' not in st.session_state:
    st.session_state.cell_res_data = []

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    model = load_model()
    
    try:
        analyzer = PatientAnalyzer(df=data)
        profile = analyzer.get_patient_profile(data)
        # Fetch individual resistance levels for all tumor cells
        st.session_state.cell_res_data = analyzer.get_cell_resistance_data() 
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if st.button("Generate Optimized Treatment Plan"):
        if model:
            env = CancerSimulation(profile)
            obs, _ = env.reset()
            history = []
            
            curr_sz, curr_ra, curr_rb = obs[0], obs[1], obs[2]
            
            for day in range(1, 31):
                # 1. Before State (Captures duplication/growth from previous day)
                history.append({
                    "Day": day,
                    "Status": "Before Drug",
                    "Action": "—",
                    "Tumor Size": int(curr_sz),
                    "Resist A": float(curr_ra),
                    "Resist B": float(curr_rb)
                })
                
                # 2. Apply Drug
                action, _ = model.predict(obs, deterministic=True)
                act_int = int(action.item())
                act_map = {0: "Rest", 1: "Drug A", 2: "Drug B"}
                
                obs, reward, terminated, truncated, info = env.step(act_int)
                
                # Update states for "After" and next "Before"
                curr_sz, curr_ra, curr_rb = obs[0], obs[1], obs[2]
                
                history.append({
                    "Day": day,
                    "Status": f"After {act_map[act_int]}",
                    "Action": act_map[act_int],
                    "Tumor Size": int(curr_sz),
                    "Resist A": float(curr_ra),
                    "Resist B": float(curr_rb)
                })
                
                if curr_sz <= 0 or terminated: break
            
            st.session_state.treatment_history = history

    if st.session_state.treatment_history:
        df_hist = pd.DataFrame(st.session_state.treatment_history)

        # --- VISUALIZATION SECTION ---
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        day_to_show = st.slider("Select Day to Visualize", 1, int(df_hist['Day'].max()), 1)
        
        day_data = df_hist[df_hist['Day'] == day_to_show]
        b_row = day_data[day_data['Status'] == "Before Drug"].iloc[0]
        a_row = day_data[day_data['Status'].str.contains("After")].iloc[0]

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("<p style='text-align:center; color:#888888;'>Before Treatment</p>", unsafe_allow_html=True)
            # Pass the individual cell resistance list to the visualization
            st.plotly_chart(create_tumor_visualization(b_row["Tumor Size"], st.session_state.cell_res_data), use_container_width=True)
        with v_col2:
            st.markdown(f"<p style='text-align:center; color:#888888;'>After {a_row['Action']}</p>", unsafe_allow_html=True)
            st.plotly_chart(create_tumor_visualization(a_row["Tumor Size"], st.session_state.cell_res_data), use_container_width=True)

        # --- LOG TABLE ---
        st.markdown("---")
        st.subheader("📊 Detailed Treatment & Evolution Log")
        
        # Formatting Resist A and B to 2 decimal places
        formatted_df = df_hist.copy()
        formatted_df["Resist A"] = formatted_df["Resist A"].map(lambda x: f"{x:.2f}")
        formatted_df["Resist B"] = formatted_df["Resist B"].map(lambda x: f"{x:.2f}")
        
        # Apply slight background color to separate Before/After visually
        def style_rows(row):
            if "Before" in row["Status"]:
                return ['background-color: #1b212c'] * len(row)
            return [''] * len(row)

        st.dataframe(formatted_df.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True)

        # --- BAR CHART SECTION ---
        st.markdown("---")
        st.subheader("📈 Tumor Size Dynamics")
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0E1117')
        ax.set_facecolor('#161B22')
        
        days = df_hist[df_hist['Status'] == "Before Drug"]['Day']
        b_vals = df_hist[df_hist['Status'] == "Before Drug"]['Tumor Size']
        a_vals = df_hist[df_hist['Status'].str.contains("After")]['Tumor Size']
        
        x = np.arange(len(days))
        ax.bar(x - 0.2, b_vals, 0.4, label='Before Drug', color='#E74C3C', alpha=0.7)
        ax.bar(x + 0.2, a_vals, 0.4, label='After Drug', color='#2ECC71', alpha=0.9)
        
        # Labels and Axis Setup
        ax.set_xlabel('Timeline (Treatment Days)', color='#C9D1D9', fontsize=10)
        ax.set_ylabel('Cell Count (Tumor Size)', color='#C9D1D9', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(days, color='#C9D1D9')
        ax.tick_params(axis='y', labelcolor='#C9D1D9')
        
        # Removing spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')

        ax.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')
        ax.grid(axis='y', alpha=0.1)
        
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Select a data source in the sidebar to begin.")