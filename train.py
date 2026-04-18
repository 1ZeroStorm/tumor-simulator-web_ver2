import pandas as pd
from stable_baselines3 import PPO
from environment import CancerEnv
from analyzer import CancerAnalyzer
import os

def train_peacekeeper_model():
    print("--- Phase 1: Diagnostic Profiling ---")
    
    # 1. Load the dataset
    try:
        data = pd.read_csv("Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Please ensure the dataset is in the same folder.")
        return

    # 2. Extract the patient profile using your Analyzer (Neural Network logic)
    analyzer = CancerAnalyzer()
    profile = analyzer.get_patient_profile(data)
    
    print(f"Targeting Patient Profile:")
    print(f" > Max Resistance A: {profile['max_res_a']:.2f}")
    print(f" > Growth Rate: {profile['avg_growth']:.2f}%")
    print("-" * 40)

    # 3. Initialize the Environment
    print("--- Phase 2: Training Strategy (Reinforcement Learning) ---")
    env = CancerEnv(profile)

    # 4. Define the PPO Agent
    # We use MlpPolicy (Multi-layer Perceptron) because our state is simple numbers.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        gamma=0.99, # Focus on long-term survival
        device="auto"
    )

    # 5. Train the Model
    # 10,000 timesteps is usually enough for this environment to find the 'Trap'
    training_steps = 10000 
    print(f"Training for {training_steps} steps...")
    model.learn(total_timesteps=training_steps)

    # 6. Save the Model
    model_name = "ppo_cancer_policy"
    model.save(model_name)
    
    print("-" * 40)
    print(f"SUCCESS: Model saved as '{model_name}.zip'")
    print("You can now run 'streamlit run app.py' to deploy the treatment tool.")

if __name__ == "__main__":
    train_peacekeeper_model()