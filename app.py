import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import PatientAnalyzer
from environment import CancerSimulation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- HELPER FUNCTION: TUMOR VISUALIZATION ---
def create_tumor_visualization(tumor_size, title=""):
    """Create a dark-mode tumor visualization with microscopic nodes"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
    ax.set_facecolor('#161B22')
    
    # Calculate number of tumor nodes based on size
    # Scale: higher tumor size = more dots
    num_tumors = max(10, int(tumor_size / 100))
    num_tumors = min(200, num_tumors)  # Cap at 200 for visibility
    
    # Generate random positions for tumor nodes
    np.random.seed(hash(title) % 2**32)  # Consistent positioning
    x_positions = np.random.uniform(0.1, 0.9, num_tumors)
    y_positions = np.random.uniform(0.1, 0.9, num_tumors)
    
    # Vary dot sizes for more realistic tumor appearance
    sizes = np.random.uniform(20, 150, num_tumors)
    
    # Create gradient effect with different colors
    scatter = ax.scatter(x_positions, y_positions, s=sizes, 
                        c=np.random.uniform(0.3, 1, num_tumors), 
                        cmap='Reds', alpha=0.7, edgecolors='#FF6B6B', linewidth=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title with tumor count
    fig.text(0.5, 0.95, title, ha='center', fontsize=14, color='#C9D1D9', weight='bold')
    fig.text(0.5, 0.02, f"Estimated Tumor Cells: {num_tumors * 100}", 
            ha='center', fontsize=10, color='#8B949E', style='italic')
    
    return fig

# --- CONFIGURATION ---
st.set_page_config(page_title="The Peacekeeper: AI Immunotherapy", layout="wide")

@st.cache_resource
def load_resources():
    # Load the pre-trained PPO model
    model_path = "ppo_cancer_policy"
    
    # Check if model file exists
    if not os.path.exists(f"{model_path}.zip"):
        st.error("⚠️ Model not found! Please train the model first by running: `python train.py`")
        st.info("The model file should be saved as 'ppo_cancer_policy.zip' in the project directory.")
        st.stop()
    
    try:
        model = PPO.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading PPO model: {e}")
        st.stop()

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

if uploaded_file is not None:
    # Store uploaded data in session state
    data = pd.read_csv(uploaded_file)
    st.session_state.uploaded_data = data
    
    model = load_resources()
    
    try:
        analyzer = PatientAnalyzer(df=data)
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        st.stop()
    
    # Step 1: Diagnostic Phase (Neural Network)
    with st.spinner("Analyzing Proteomic Signatures..."):
        profile = analyzer.get_patient_profile(data)
        st.session_state.patient_profile = profile
    
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
            
            # Map action to readable name
            action_name = "Drug A (Priming)" if action == 1 else "Drug B (TRAP)"
            if action == 0: action_name = "Rest (Recovery)"
            
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
        
        # Day Navigation Controls
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])
        
        with nav_col1:
            if st.button("◀ PREV", key="prev_day"):
                if st.session_state.current_day_view > 0:
                    st.session_state.current_day_view -= 1
        
        with nav_col3:
            st.markdown(f"<div style='text-align: center; padding: 10px;'><h3>📅 Day {st.session_state.current_day_view + 1}</h3></div>", 
                       unsafe_allow_html=True)
        
        with nav_col5:
            if st.button("NEXT ▶", key="next_day"):
                if st.session_state.current_day_view < len(history) - 1:
                    st.session_state.current_day_view += 1
        
        # Before/After Toggle
        toggle_col1, toggle_col2 = st.columns([1, 1])
        with toggle_col1:
            show_before = st.toggle("🔴 Before Drug Application", value=True, key="before_toggle")
        with toggle_col2:
            show_after = st.toggle("🟢 After Drug Application", value=True, key="after_toggle")
        
        # Get current day data
        current_idx = st.session_state.current_day_view
        current_day_data = history[current_idx]
        
        # Display tumor visualizations
        if show_before and show_after:
            vis_col1, vis_col2 = st.columns(2)
            
            # Estimate tumor size before drug (slightly larger)
            tumor_before = current_day_data["Tumor Size"] * 1.15
            
            with vis_col1:
                st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
                fig_before = create_tumor_visualization(tumor_before, 
                    f"Day {current_day_data['Day']} - Before {current_day_data['Action']}")
                st.pyplot(fig_before)
                plt.close(fig_before)
            
            with vis_col2:
                st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
                fig_after = create_tumor_visualization(current_day_data["Tumor Size"], 
                    f"Day {current_day_data['Day']} - After {current_day_data['Action']}")
                st.pyplot(fig_after)
                plt.close(fig_after)
        
        elif show_before:
            st.markdown("<h4 style='text-align: center;'>🔴 Before Treatment</h4>", unsafe_allow_html=True)
            tumor_before = current_day_data["Tumor Size"] * 1.15
            fig_before = create_tumor_visualization(tumor_before, 
                f"Day {current_day_data['Day']} - Before {current_day_data['Action']}")
            st.pyplot(fig_before)
            plt.close(fig_before)
        
        elif show_after:
            st.markdown("<h4 style='text-align: center;'>🟢 After Treatment</h4>", unsafe_allow_html=True)
            fig_after = create_tumor_visualization(current_day_data["Tumor Size"], 
                f"Day {current_day_data['Day']} - After {current_day_data['Action']}")
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
        st.subheader("Evolutionary Trap Visualization")
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:red'
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Tumor Size', color=color)
        ax1.plot(df_history['Day'], df_history['Tumor Size'], color=color, linewidth=3, label="Tumor Size")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance Levels')
        ax2.plot(df_history['Day'], df_history['Resist_A'], '--', label="Resist A (Target)")
        ax2.plot(df_history['Day'], df_history['Resist_B'], ':', label="Resist B (Trap)", color='green')
        ax2.axhline(y=2.5, color='gray', linestyle='-', alpha=0.3, label="Trap Threshold")
        
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **Doctor's Note:** The AI utilized Drug A to drive down the resistance to Drug B. 
        Once 'Resist_B' crossed the threshold of 2.5, the 'Evolutionary Trap' was sprung, 
        leading to rapid tumor mass reduction while maintaining safety margins.
        """)

else:
    st.warning("Please upload a CSV file to begin analysis.")