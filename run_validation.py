import os
import time
from stable_baselines3 import PPO
from environment import CancerSimulation
from analyzer import PatientAnalyzer

def logger(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open("simulation_test.log", "a") as f:
        f.write(log_entry + "\n")

def run_test():
    # Clear old log
    if os.path.exists("simulation_test.log"):
        os.remove("simulation_test.log")
    
    logger("--- STARTING SYSTEM VALIDATION ---")

    # 1. Check for Data
    csv_path = "data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"
    if os.path.exists(csv_path):
        logger(f"SUCCESS: Dataset found at {csv_path}")
        analyzer = PatientAnalyzer(csv_path)
        profile = analyzer.get_strategic_profile()
        logger(f"Patient Profile Loaded: {profile}")
    else:
        logger("ERROR: Dataset missing in /data folder!")
        return

    # 2. Check for Model
    model_path = "peacekeeper_final_azure.zip"
    if os.path.exists(model_path):
        logger(f"SUCCESS: Azure Model found: {model_path}")
        try:
            model = PPO.load("peacekeeper_final_azure")
            logger("SUCCESS: Model loaded into memory via Stable-Baselines3")
        except Exception as e:
            logger(f"ERROR: Model loading failed: {e}")
            return
    else:
        logger("ERROR: peacekeeper_final_azure.zip not found!")
        return

    # 3. Run Simulation Loop (10 Days)
    logger("--- STARTING 10-DAY SIMULATION RUN ---")
    env = CancerSimulation(profile)
    obs, _ = env.reset()
    
    for day in range(1, 11):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action mapping for the log
        action_name = ["No Treatment", "Drug A (Priming)", "Drug B (Trap)"][action]
        
        status = (f"Day {day} | Action: {action_name} | "
                  f"Size: {obs[0]:.2f} | Res_A: {obs[1]:.2f} | Res_B: {obs[2]:.2f}")
        logger(status)
        
        if terminated:
            logger("Simulation reached termination state.")
            break

    logger("--- VALIDATION COMPLETE: LOG SAVED TO simulation_test.log ---")

if __name__ == "__main__":
    run_test()