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
        size, res_a, res_b, toxicity = self.state
        # size = 1000      
        # res_a = currently the average resistance of all tumor
        # res_b = initialized 5.0
        # toxicity = current toxicity level
        self.day += 1
        # initialize from self.reset() and adding first day

        size *= (1.0 + self.profile['avg_growth'] / 100.0)
        # average growth size increment
        
        reward = 0
        
        # Toxicology: Update consecutive drugs and toxicity
        if action == 1 or action == 2:  # Drug use
            self.consecutive_drugs += 1
            self.toxicity += 1.0  # Increase toxicity
            if self.consecutive_drugs >= 5:
                reward -= 1000  # Massive penalty for 5+ consecutive drug days
        else:  # Rest
            self.consecutive_drugs = 0
            self.toxicity = max(0.0, self.toxicity - 0.5)  # Recover toxicity
        
        if action == 1: # Drug A (Standard)
            kill_rate = max(0.05, 0.9 - (res_a / self.profile['max_res_a']))
            # The 0.9 represents a drug that is 90% effective under perfect conditions (when the cancer has zero resistance).
            # The 0.05 represents the 5% minimum kill rate. This is the "Floor."
                #Why is it there? Without this, if the cancer's resistance (res_a) became equal to the max_res_a, the math would result in 0% kill rate.
            
            size -= (size * kill_rate) # reducing size

            res_a += 0.3 # Mutation
            res_b -= 0.4 # Collateral Sensitivity (The Trap)
            reward = 10

        elif action == 2: # Drug B (The Trap)
            kill_rate = 0.85 if res_b < 2.5 else 0.05
            # By setting it at 0.85, the AI has to decide: "Do I use the hammer once? Or do I need to use it multiple times to finish the job?"

            size -= (size * kill_rate)
            res_b += 0.5
            reward = 100 if kill_rate > 0.8 else -20

        # why use drug A and B instead of real drugs
        '''
        In physics, you learn about "frictionless surfaces" or "ideal gases." 
        In AI medicine, we use "Drug A" to represent a Class of Action.

        Drug A represents any drug that targets a 
        specific protein (like an Oncogene).

        Drug B represents the "Counter-attack" drug.
        By keeping the names general, your code is actually more powerful. 
        It means your AI could be used for any two drugs that have a 
        "Trade-off" (Collateral Sensitivity) relationship, not just one specific chemical.
        '''

        reward -= (size * 0.1)
        self.state = np.array([size, res_a, res_b, self.toxicity], dtype=np.float32)
        
        done = bool(size < 1 or self.day >= 60)
        # side is small enough or timeout

        if size < 1: reward += 1000
        return self.state, reward, done, False, {}