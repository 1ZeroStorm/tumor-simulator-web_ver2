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
    return None

st.title("🛡️ OncoSteer")
st.markdown("### Steering tumor evolution toward therapeutic vulnerability")

# --- SIDEBAR: DATA SELECTION ---
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
        st.sidebar.error(f"Default data file not found.")

# Initialize session state
if 'treatment_history' not in st.session_state:
    st.session_state.treatment_history = None
if 'current_day_view' not in st.session_state:
    st.session_state.current_day_view = 0
if 'last_loaded_file' not in st.session_state:
    st.session_state.last_loaded_file = None

if uploaded_file is not None:
    if current_data_source != st.session_state.last_loaded_file:
        st.session_state.treatment_history = None
        st.session_state.current_day_view = 0
        st.session_state.last_loaded_file = current_data_source

    data = pd.read_csv(uploaded_file)
    model = load_model()
    
    try:
        analyzer = PatientAnalyzer(df=data)
        profile = analyzer.get_patient_profile(data)
        st.session_state.cell_resistance_data = analyzer.get_cell_resistance_data()
        st.session_state.patient_profile = profile
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if st.button("Generate Optimized Treatment Plan"):
        if model:
            env = CancerSimulation(profile)
            obs, _ = env.reset()
            history = []
            
            # Initial state tracking for the "Before" part of Day 1
            current_size = obs[0]
            current_res_a = obs[1]
            current_res_b = obs[2]
            
            for day in range(1, 31):
                # Before Drug Application
                day_entry = {
                    "Day": day,
                    "Action": "",
                    "Status": "Before Drug",
                    "Tumor Size": int(current_size),
                    "Resist A": round(float(current_res_a), 2),
                    "Resist B": round(float(current_res_b), 2)
                }
                
                # Predict and Step
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action.item())
                action_names = {0: "Rest", 1: "Drug A", 2: "Drug B"}
                
                obs, reward, terminated, truncated, info = env.step(action_int)
                
                # After Drug Application
                current_size = obs[0]
                current_res_a = obs[1]
                current_res_b = obs[2]
                
                after_entry = {
                    "Day": day,
                    "Action": action_names.get(action_int, "N/A"),
                    "Status": f"After {action_names.get(action_int, 'N/A')}",
                    "Tumor Size": int(current_size),
                    "Resist A": round(float(current_res_a), 2),
                    "Resist B": round(float(current_res_b), 2)
                }
                
                history.append(day_entry)
                history.append(after_entry)
                
                if current_size <= 0 or terminated:
                    break
            
            st.session_state.treatment_history = history

    if st.session_state.treatment_history:
        history = st.session_state.treatment_history
        df_hist = pd.DataFrame(history)

        # Visualizations
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        
        # Navigation
        max_days = df_hist['Day'].max()
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("◀ PREV") and st.session_state.current_day_view > 0:
                st.session_state.current_day_view -= 1
                st.rerun()
        with nav_col2:
            st.markdown(f"<h3 style='text-align: center;'>📅 Day {st.session_state.current_day_view + 1}</h3>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("NEXT ▶") and st.session_state.current_day_view < max_days - 1:
                st.session_state.current_day_view += 1
                st.rerun()

        # Visualization for specific day view
        day_data = df_hist[df_hist['Day'] == (st.session_state.current_day_view + 1)]
        before_row = day_data[day_data['Status'] == "Before Drug"].iloc[0]
        after_row = day_data[day_data['Status'].str.contains("After")].iloc[0]

        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_tumor_visualization(before_row["Tumor Size"], before_row["Resist A"]), use_container_width=True)
        with vis_col2:
            st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_tumor_visualization(after_row["Tumor Size"], after_row["Resist A"]), use_container_width=True)

        # The Requested Table
        st.markdown("---")
        st.subheader("📊 Detailed Treatment & Evolution Log")
        
        # Applying some styling to differentiate rows
        def style_rows(row):
            if "Before" in row["Status"]:
                return ['background-color: #1b212c'] * len(row)
            return [''] * len(row)

        st.dataframe(df_hist.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True)

        # Bar Chart Comparison
        st.markdown("---")
        st.subheader("📈 Tumor Size Dynamics")
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0E1117')
        ax.set_facecolor('#161B22')
        
        before_vals = df_hist[df_hist['Status'] == "Before Drug"]
        after_vals = df_hist[df_hist['Status'].str.contains("After")]
        
        x = np.arange(len(before_vals))
        ax.bar(x - 0.2, before_vals['Tumor Size'], 0.4, label='Before Drug', color='#E74C3C')
        ax.bar(x + 0.2, after_vals['Tumor Size'], 0.4, label='After Drug', color='#2ECC71')
        
        ax.set_xticks(x)
        ax.set_xticklabels(before_vals['Day'], color='#C9D1D9')
        ax.legend()
        st.pyplot(fig)

else:
    st.warning("Please provide data to start.")