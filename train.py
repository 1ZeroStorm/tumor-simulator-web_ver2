import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment import CancerSimulation
from analyzer import PatientAnalyzer
import os

class TrainingOutputCallback(BaseCallback):
    """Custom callback to display clean training output"""
    def __init__(self):
        super(TrainingOutputCallback, self).__init__()
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check if episode is done
        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_count += 1
            # Only print every 10 episodes to reduce clutter
            if self.episode_count % 10 == 0:
                print(f"✓ Episode {self.episode_count} completed")
        return True

def train_peacekeeper_model():
    print("--- Phase 1: Diagnostic Profiling ---")
    
    # 1. Load the dataset
    try:
        data = pd.read_csv("data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Please ensure the dataset is in the same folder.")
        return

    # 2. Extract the patient profile using your Analyzer (Neural Network logic)
    analyzer = PatientAnalyzer(df=data)
    profile = analyzer.get_patient_profile()
    
    print(f"Targeting Patient Profile:")
    print(f" > Max Resistance A: {profile['max_res_a']:.2f}")
    print(f" > Growth Rate: {profile['avg_growth']:.2f}%")
    print("-" * 40)

    # 3. Initialize the Environment
    print("--- Phase 2: Training Strategy (Reinforcement Learning) ---")
    env = CancerSimulation(profile)

    # 4. Define the PPO Agent (verbose=0 to suppress PPO training logs)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0,  # Suppress verbose PPO output
        learning_rate=0.0003, 
        gamma=0.99,
        device="auto"
    )

    # 5. Train the Model with custom callback
    training_steps = 10000 
    print(f"Training for {training_steps} steps...\n")
    
    callback = TrainingOutputCallback()
    model.learn(total_timesteps=training_steps, callback=callback)

    # 6. Show a sample episode with desired output format
    print("\n" + "-" * 40)
    print("Sample Episode Output:")
    print("-" * 40)
    
    obs, _ = env.reset()
    done = False
    day = 1
    
    action_names = {0: "Rest", 1: "Drug A", 2: "Drug B"}
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        size, res_a, res_b, toxicity = obs
        
        action_name = action_names[action]
        print(f"Day {day}: Action={action_name} | Tumor Size={size:.0f} | Res_A={res_a:.2f} | Res_B={res_b:.2f} | Toxicity={toxicity:.2f}")
        
        day += 1
    
    # 7. Save the Model
    model_name = "ppo_cancer_policy"
    model.save(model_name)
    
    print("\n" + "-" * 40)
    print(f"SUCCESS: Model saved as '{model_name}.zip'")
    print("You can now run 'streamlit run app.py' to deploy the treatment tool.")

if __name__ == "__main__":
    train_peacekeeper_model()