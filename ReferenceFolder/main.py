from analyzer import PatientAnalyzer
from environment import CancerSimulation
from stable_baselines3 import PPO

def execute_clinical_pipeline():
    # 1. DIAGNOSIS (Using DiagnosticNet)
    print("--- Phase 1: Neural Network Diagnosis ---")
    analyzer = PatientAnalyzer('Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv')
    
    strategic_profile = analyzer.get_strategic_profile()
    '''
    return {
            "avg_growth": number
            "max_res_a": number
            "starting_res_a": number
        }
    '''

    print(f"NN detected max resistance of: {strategic_profile['max_res_a']:.2f}")
    print(f"NN detected growth rate of: {strategic_profile['avg_growth']:.2f}")

    # 2. TREATMENT OPTIMIZATION (Using PPO)
    print("\n--- Phase 2: AI Strategy Optimization ---")
    env = CancerSimulation(strategic_profile)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0005)
    model.learn(total_timesteps=100000)

    # 3. VERIFICATION (Testing the Trap)
    obs, _ = env.reset()
    print("\n--- Phase 3: Optimized Treatment Plan ---")
    for d in range(1, 61):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
        action_name = ["Rest", "Drug A", "Drug B (TRAP)"][action]
        tumor_size = int(obs[0])
        res_a = obs[1]
        res_b = obs[2]
        toxicity = obs[3]
        print(
            f"Day {d}: Action={action_name} | Tumor Size={tumor_size} | "
            f"Res_A={res_a:.2f} | Res_B={res_b:.2f} | Toxicity={toxicity:.2f}"
        )
        
        if done:
            status = "CURED" if obs[0] < 1 else "FAILED"
            print(f"\nResult: {status} at Day {d}")
            break

if __name__ == "__main__":
    execute_clinical_pipeline()