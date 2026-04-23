import gymnasium as gym
import numpy as np

class CancerSimulation(gym.Env):
    def __init__(self, profile):
        super(CancerSimulation, self).__init__()
        self.profile = profile

        self.observation_space = gym.spaces.Box(low=0, high=1e6, shape=(4,), dtype=np.float32)
        # defines 1D array with 4 elements (ex: [speed, distance, temperature, toxicity])
        # each elements can't surpass 1 million
        # lowest value is 0 (can't be negative)
        # float 32 bit for standard neural network 

        self.action_space = gym.spaces.Discrete(3)
        # 3 fixed moves (0, 1, 2) (ex: 0: left, 1: straight, 2: right) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # takes the seed you provided and re-seeds self.np_random
        # seed = None -> generates a random seed for you

        # State: [Size, Res_A, Res_B, Toxicity]
        self.state = np.array([1000.0, self.profile['starting_res_a'], 5.0, 0.0], dtype=np.float32)
        # 1000 -> inital tumor size
        # initial resistance for drug A
        # 5.0 -> initial resistance for drug B
        # 0.0 -> initial toxicity

        self.day = 0
        self.consecutive_drugs = 0
        self.toxicity = 0.0
        return self.state, {}

    def step(self, action):
        size_before, res_a, res_b, toxicity_before = self.state
        self.day += 1
        
        # 1. Natural Tumor Growth
        size = size_before * (1.0 + self.profile['avg_growth'] / 100.0)
        
        # 2. Toxicity Logic (Keep your existing constraints)
        if action == 1 or action == 2:
            self.consecutive_drugs += 1
            self.toxicity += 1.0
        else:
            self.consecutive_drugs = 0
            self.toxicity = max(0.0, self.toxicity - 0.5)

        # 3. Drug Effects
        if action == 1: # Drug A (Standard)
            kill_rate = max(0.05, 0.9 - (res_a / self.profile['max_res_a']))
            size -= (size * kill_rate)
            res_a += 0.3
            res_b -= 0.4 # The Trap: Making the cancer weak to Drug B
        
        elif action == 2: # Drug B (The Trap)
            kill_rate = 0.85 if res_b < 2.5 else 0.05
            size -= (size * kill_rate)
            res_b += 0.5

        # 4. NEW REWARD LOGIC (The "Peacekeeper" Pivot)
        reward = 0
        
        # A. Shrinkage Bonus: Reward the AI for the AMOUNT it shrinks the tumor
        shrinkage = size_before - size
        if shrinkage > 0:
            reward += (shrinkage * 0.5)  # 0.5 points for every unit killed
        else:
            reward -= (abs(shrinkage) * 1.0) # Heavier penalty for letting it grow

        # B. Toxicity Penalties (Keeping it alive)
        if self.consecutive_drugs >= 5:
            reward -= 500  # Penalty for pushing the patient too hard
        
        reward -= (self.toxicity * 2.0) # Constant small "nudge" to keep toxicity low

        # C. Strategic "Trap" Bonus (Optional but helpful)
        # Give a tiny "hint" when the AI successfully lowers Resistance B
        if action == 1 and res_b < 5.0:
            reward += 5 

        # 5. Termination Logic
        self.state = np.array([size, res_a, res_b, self.toxicity], dtype=np.float32)
        
        done = bool(size < 1 or self.day >= 60 or self.toxicity > 10.0)
        
        if size < 1: 
            reward += 2000 # Massive jackpot for curing the cancer
        elif self.day >= 60:
            reward -= 500 # Penalty for failing to cure in time
            
        return self.state, reward, done, False, {}