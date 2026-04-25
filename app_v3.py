import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer_v3 import PatientAnalyzer
from environment_v3 import CancerSimulation
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
import os

# --- HELPER FUNCTION: TUMOR VISUALIZATION ---
def create_tumor_visualization(tumor_size, resistance_list, max_res=15.0): 
    # tumor size = 1000
    # resistance list: a list containing resistance level or all cells (ex: 2000 since we disinclude healthy tumors)

    """
    High-performance tumor cell visualization using drawable canvas.
    Maps individual resistance values to colors for every single cell.
    """
    if 'cell_coordinates' not in st.session_state:

        # Persistent coordinate pool to keep cell positions stable during refresh
        # 20000 rows and 2 cols (represents coords interval 0 to 1) 
        st.session_state.cell_coordinates = np.random.rand(20000, 2)

    num_cells = int(min(len(st.session_state.cell_coordinates), max(1, tumor_size))) # setting the num_cell to match the tumor_size
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()
    # [:num_cells] -> start at index 0 stop at index num_cells

    # 1. Handle Resistance Data
    # If we have a list of individual resistances, we slice it to match current population
    if isinstance(resistance_list, (list, np.ndarray)) and len(resistance_list) > 0: 
        # enter this block if the resistance list is an np array or list and not empty
        # We cycle or pad the list if the current tumor size exceeds our data sample
        
        if len(resistance_list) < num_cells: 
            # if resistance_list for example (300) and num_cells: 1000

            repeats = (num_cells // len(resistance_list)) + 1 # 3+1 = 4
            current_resistances = np.tile(resistance_list, repeats)[:num_cells]
            # extenting the resistance 4 times, taking num_cells elements from 0 from that list
        else:
            current_resistances = np.array(resistance_list[:num_cells])
            # just taking the resistance list, stopped in index num_cells

    else:
        # Fallback if resistance_list is not a list or is empty
        
        current_resistances = np.full(num_cells, 5.0) # create a 1D array [5,5,5,5,...num_cells times]

    # 2. Normalize for colorscale (0 to 1)
    norm_colors = np.clip(current_resistances / max_res, 0, 1)
    # current resistances 0 < x < max_res, 
    # if current_resistances / max_res surpass [0, 1] close interval, then it will be clipped 0 or 1

    # 3. Create background image with matplotlib
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0E1117')
    ax.set_facecolor('#161B22')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Plot cells with color mapping
    scatter = ax.scatter(
        cell_coords[:, 0],  # 1D array of x coords of every cell
        cell_coords[:, 1],  # 1D array of y coords of every cell
        c=norm_colors,
        cmap='YlOrRd',  # yellow= low, orange= med, red = high
        s=20,
        alpha=0.8,  # 80% solid, 20% transparent
        vmin=0,  # clipping the colors, checking norm_colors if theres below 0, keep the same color as 0
        vmax=1   # this applied also to above 1
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Individual Resistance', color='#888888', fontsize=9)
    cbar.ax.tick_params(colors='#888888', labelsize=8)
    
    ax.set_title(f"Live Population: {num_cells:,} cells", color='#888888', fontsize=12, pad=10)
    
    plt.tight_layout()
    
    # Convert matplotlib figure to image for canvas
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # 4. Display using drawable canvas in view mode
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=0,
        background_image=image,
        height=600,
        width=600,
        drawing_mode="view",
        key=f"canvas_{num_cells}"
    )
    
    return canvas_result

# --- CONFIGURATION ---
st.set_page_config( 
    page_title="OncoSteer: Evolutionary AI", # your browser bar on top
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "peacekeeper_final_azure" 
DEFAULT_DATA_PATH = "data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"

@st.cache_resource # decorate function load_model() to just run once even when the web refreshes
def load_model(): # returns the PPO model
    if os.path.exists(f"{MODEL_PATH}.zip"):
        return PPO.load(MODEL_PATH)
    return None

st.title("🧬 OncoSteer")
st.markdown("### Steering Tumor Evolution Toward Therapeutic Vulnerability.")

# --- SIDEBAR ---
st.sidebar.header("Patient Data Input")
data_mode = st.sidebar.radio("Select Data Source:", ["Upload CSV", "Use Default Synthetic Data"])

uploaded_file = None
current_data_source = None

#input csv, stored in uploaded_file
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

# treatment_history and cell_res_data attribute in st.session_state
if 'treatment_history' not in st.session_state:
    st.session_state.treatment_history = None
if 'cell_res_data' not in st.session_state:
    st.session_state.cell_res_data = []

# check if uploaded_file is not empty
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file) # pandas csv of data
    model = load_model() # loading the PPO model from azure
    
    try:
        analyzer = PatientAnalyzer(df=data) # analyzer tool, to do/return stuff from the csv
        profile = analyzer.get_patient_profile(data) # return avg resistance, avg growth rate
        # Fetch individual resistance levels for all tumor cells
        st.session_state.cell_res_data = analyzer.get_cell_resistance_data()  # returns 1D array list of 2000+ resistance levels of every cell   
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if st.button("Generate Optimized Treatment Plan"):
        if model:
            env = CancerSimulation(profile)
            obs, _ = env.reset()
            history = []
            
            curr_sz, curr_ra, curr_rb = obs[0], obs[1], obs[2]
            # curr_sz -> current population
            # curr_ra -> current resistance A
            # curr_rb -> current resistance B

            for day in range(1, 31):
                # 1. Before State (Captures duplication/growth from previous day)
                history.append({
                    "Day": day,
                    "Status": "Before Drug (Prior Duplication)",
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
                
                # Get size after duplication from info dictionary
                sz_after_duplication = info['size_after_duplication']
                
                # Add history entry for After Duplication
                history.append({
                    "Day": day,
                    "Status": "Before Drug (Post Duplication)",
                    "Action": "—",
                    "Tumor Size": int(sz_after_duplication),
                    "Resist A": float(curr_ra),
                    "Resist B": float(curr_rb)
                })
                
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
        
        #adding slider
        st.markdown("---")
        st.subheader("🔬 Microscopic Tumor Evolution")
        day_to_show = st.slider("Select Day to Visualize", 1, int(df_hist['Day'].max()), 1)
        


        # 
        day_data = df_hist[df_hist['Day'] == day_to_show]
        c_row = day_data[day_data['Status'] == "Before Drug (Prior Duplication)"].iloc[0]
        b_row = day_data[day_data['Status'] == "Before Drug (Post Duplication)"].iloc[0]
        a_row = day_data[day_data['Status'].str.contains("After")].iloc[0]

        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col3:
            st.markdown("<p style='text-align:center; color:#888888;'>Before Drug (Prior Duplication)</p>", unsafe_allow_html=True)
            create_tumor_visualization(c_row["Tumor Size"], st.session_state.cell_res_data)
        with v_col2:
            st.markdown("<p style='text-align:center; color:#888888;'>Before Drug (Post Duplication)</p>", unsafe_allow_html=True)
            create_tumor_visualization(b_row["Tumor Size"], st.session_state.cell_res_data)
        with v_col3:
            st.markdown(f"<p style='text-align:center; color:#888888;'>After {a_row['Action']}</p>", unsafe_allow_html=True)
            create_tumor_visualization(a_row["Tumor Size"], st.session_state.cell_res_data)

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
        
        days = df_hist[df_hist['Status'] == "Before Drug (Prior Duplication)"]['Day']
        b_vals = df_hist[df_hist['Status'] == "Before Drug (Prior Duplication)"]['Tumor Size']
        dup_vals = df_hist[df_hist['Status'] == "Before Drug (Post Duplication)"]['Tumor Size']
        a_vals = df_hist[df_hist['Status'].str.contains("After")]['Tumor Size']
        
        x = np.arange(len(days))
        ax.bar(x - 0.3, b_vals, 0.25, label='Before Duplication', color='#E74C3C', alpha=0.7)
        ax.bar(x - 0.05, dup_vals, 0.25, label='After Duplication', color='#F39C12', alpha=0.8)
        ax.bar(x + 0.2, a_vals, 0.25, label='After Drug', color='#2ECC71', alpha=0.9)
        
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