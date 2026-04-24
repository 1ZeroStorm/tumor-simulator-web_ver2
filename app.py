import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import PatientAnalyzer
from environment import CancerSimulation
import matplotlib.pyplot as plt
import os
import time

# --- CONFIGURATION & PAGE STYLE ---
st.set_page_config(page_title="Peacekeeper AI | Cancer Strategy", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #C9D1D9; }
    stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #238636; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL & DATA LOADING ---
MODEL_PATH = "peacekeeper_final_azure" # SB3 adds .zip automatically
DATA_PATH = "data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"

@st.cache_resource
def load_model():
    if os.path.exists(f"{MODEL_PATH}.zip"):
        return PPO.load(MODEL_PATH, device="cpu")
    return None

@st.cache_data
def get_patient_data():
    if os.path.exists(DATA_PATH):
        analyzer = PatientAnalyzer(DATA_PATH)
        return analyzer.get_strategic_profile()
    return {"avg_growth": 14.0, "max_res_a": 15.0, "starting_res_a": 9.0}

model = load_model()
patient_profile = get_patient_data()

# --- SESSION STATE MANAGEMENT ---
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.history = []
    st.session_state.current_size = 1000.0
    # Initialize 5000 persistent coordinates for the "jelly" effect
    st.session_state.cell_coords = np.random.rand(5000, 2) 
    st.session_state.last_display_size = 1000.0

# --- VISUALIZATION FUNCTION ---
def render_tumor_visual(display_size, res_a, max_res=15.0):
    """Renders cells as dots. Redness increases with Res_A."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0E1117')
    ax.set_facecolor('#161B22')
    
    # Scale tumor size to number of dots (Max 5000 for performance)
    num_dots = int(np.clip(display_size, 10, 5000))
    coords = st.session_state.cell_coords[:num_dots]
    
    # Calculate Redness based on Resistance A
    # 0 Res = Blue/Green, Max Res = Bright Red
    norm_res = np.clip(res_a / max_res, 0, 1)
    cell_color = (norm_res, 0.2, 1 - norm_res, 0.6) # RGBA
    
    ax.scatter(coords[:, 0], coords[:, 1], s=12, c=[cell_color], edgecolors='none')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

# --- MAIN UI ---
st.title("🧬 Peacekeeper AI: Evolutionary Trap Simulation")
st.sidebar.header("Patient Profile (from Data)")
st.sidebar.write(f"Avg Growth: {patient_profile['avg_growth']:.2f}")
st.sidebar.write(f"Initial Res A: {patient_profile['starting_res_a']:.2f}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Microscopic Tumor Environment")
    tumor_placeholder = st.empty()
    
    # Display current tumor state
    # If it's the first run, it shows the initial 1000 cells
    initial_res = st.session_state.history[-1]['Res_A'] if st.session_state.history else patient_profile['starting_res_a']
    fig = render_tumor_visual(st.session_state.last_display_size, initial_res)
    tumor_placeholder.pyplot(fig)

with col2:
    st.subheader("Simulation Control")
    
    if st.button("Run Strategic Next Step"):
        if model:
            # 1. Setup environment state
            # State: [Size, Res_A, Res_B, Toxicity]
            res_a_curr = st.session_state.history[-1]['Res_A'] if st.session_state.history else patient_profile['starting_res_a']
            res_b_curr = st.session_state.history[-1]['Res_B'] if st.session_state.history else 5.0
            tox_curr = st.session_state.history[-1]['Toxicity'] if st.session_state.history else 0.0
            
            obs = np.array([st.session_state.current_size, res_a_curr, res_b_curr, tox_curr], dtype=np.float32)
            
            # 2. AI Predicts Action
            action, _ = model.predict(obs, deterministic=True)
            
            # 3. Step the simulation (Logic inside environment.py)
            env = CancerSimulation(patient_profile)
            env.state = obs # Sync state
            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # 4. "Bloop" Animation Logic
            target_size = new_obs[0]
            start_size = st.session_state.last_display_size
            
            # Animate growth/shrinkage in 5 frames
            for i in range(1, 6):
                temp_size = start_size + (target_size - start_size) * (i / 5)
                with tumor_placeholder.container():
                    fig_anim = render_tumor_visual(temp_size, new_obs[1])
                    st.pyplot(fig_anim)
                time.sleep(0.05)
            
            # 5. Update state
            st.session_state.current_size = new_obs[0]
            st.session_state.last_display_size = new_obs[0]
            st.session_state.step += 1
            
            action_map = {0: "Wait", 1: "Drug A (Priming)", 2: "Drug B (Trap)"}
            st.session_state.history.append({
                "Day": st.session_state.step,
                "Action": action_map[action],
                "Size": new_obs[0],
                "Res_A": new_obs[1],
                "Res_B": new_obs[2],
                "Toxicity": new_obs[3]
            })

    # Show Progress Table
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.table(df_history.tail(5))

# --- PLOTS ---
if st.session_state.history:
    st.divider()
    st.subheader("Strategic Analytics")
    chart_data = pd.DataFrame(st.session_state.history).set_index("Day")
    st.line_chart(chart_data[["Size", "Res_A", "Res_B"]])

if not model:
    st.error("Model file not found. Please ensure peacekeeper_final_azure.zip is in the root folder.")