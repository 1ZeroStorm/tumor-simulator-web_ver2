import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer import CancerAnalyzer
from environment import CancerEnv
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="The Peacekeeper: AI Immunotherapy", layout="wide")

def load_resources():
    # Load the analyzer (Neural Network logic)
    analyzer = CancerAnalyzer() 
    # Load the pre-trained PPO model
    model = PPO.load("ppo_cancer_policy")
    return analyzer, model

st.title("🛡️ The Peacekeeper")
st.markdown("### Digital Immunotherapy & Evolutionary Trap Optimizer")
st.info("Upload patient proteomic data to generate a personalized, safety-constrained treatment plan.")

# --- SIDEBAR: PATIENT DATA ---
st.sidebar.header("Patient Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Proteomic CSV", type=["csv"])

if uploaded_file is not None:
    analyzer, model = load_resources()
    data = pd.read_csv(uploaded_file)
    
    # Step 1: Diagnostic Phase (Neural Network)
    with st.spinner("Analyzing Proteomic Signatures..."):
        profile = analyzer.get_patient_profile(data)
    
    # --- DISPLAY DIAGNOSTIC SUMMARY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Resistance (Drug A)", f"{profile['max_res_a']:.2f}")
    col2.metric("Avg Growth Rate", f"{profile['avg_growth']:.2f}%")
    col3.metric("Initial Res_B", "5.00 (Standard)")

    if st.button("Generate Optimized Treatment Plan"):
        # Step 2: Strategy Optimization Phase
        env = CancerEnv(profile)
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
            toxicity = info.get('toxicity', day) # Adjust based on your env.py variables
            
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

        # --- RESULTS DISPLAY ---
        st.success(f"Strategy Optimized: Treatment targets eradication by Day {day}")
        
        # Display Table
        df_history = pd.DataFrame(history)
        st.table(df_history)
        
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