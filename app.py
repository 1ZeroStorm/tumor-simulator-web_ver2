import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import CancerSimulation
from analyzer import PatientAnalyzer

st.set_page_config(page_title="AI Tumor Simulator", layout="wide")

st.title("🧬 AI Cancer Treatment Simulator")
st.write("Simulate tumor growth and AI-driven drug applications in real-time.")

# --- Sidebar for Data Input ---
st.sidebar.header("Data Source")
data_option = st.sidebar.radio("Select Data:", ["Use Default Data", "Upload My Own CSV"])

csv_input = None # Initialize

if data_option == "Use Default Data":
    csv_input = "data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"
else:
    csv_input = st.sidebar.file_uploader("Upload CSV", type="csv")

# --- DEBUG LOGGING SECTION ---
st.subheader("System Log")
st.write("Current `csv_input` variable type:", type(csv_input))
st.write("Value of `csv_input`:", csv_input)

if csv_input is not None:
    # Let's actually look at the data to be sure
    df_preview = pd.read_csv(csv_input)
    st.write("📊 Data Preview (First 5 rows):")
    st.dataframe(df_preview.head()) 
    
    # IMPORTANT: Reset the "cursor" of the file if it's an upload
    # so the Analyzer can read it again later.
    if hasattr(csv_input, 'seek'):
        csv_input.seek(0)

# --- Simulation Logic ---
if st.button("🚀 Start Simulation"):
    if csv_input is not None:
        try:
            # Pass the input (either the string path or the uploaded file object)
            analyzer = PatientAnalyzer(csv_input)
            profile = analyzer.get_strategic_profile()
            env = CancerSimulation(profile)
            
            obs, _ = env.reset()
            
            # Placeholders for the "Lively" updates
            col1, col2 = st.columns(2)
            with col1:
                status_text = st.empty()
                chart_place = st.empty()
            with col2:
                log_text = st.empty()
            
            tumor_history = []
            log_messages = []

            for day in range(1, 61):
                # Example Logic: Switch to Drug B if Resistance A is high
                action = 1 if obs[1] < 12 else 2 
                
                obs, reward, done, _, _ = env.step(action)
                
                # UI Updates
                action_name = ["Rest", "Drug A", "Drug B (TRAP)"][action]
                tumor_history.append(obs[0])
                
                status_text.metric("Current Tumor Size", f"{int(obs[0])} cells", delta=f"{int(obs[0] - (tumor_history[-2] if len(tumor_history)>1 else 1000))}")
                
                # Update Chart
                chart_place.line_chart(tumor_history)
                
                # Update Log
                log_messages.append(f"Day {day}: Applied {action_name}")
                log_text.text_area("Treatment Logs", "\n".join(log_messages[::-1]), height=300)
                
                time.sleep(0.1) # Controls the speed of the "lively" simulation
            
            st.success("Simulation Complete!")
            
        except FileNotFoundError:
            st.error(f"Error: The file '{csv_input}' was not found in the repository folder.")
    else:
        st.warning("Please upload a file or select 'Use Default Data' first.")

    