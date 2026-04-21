import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DiagnosticNet(nn.Module):
    def __init__(self, input_dim=5):
        super(DiagnosticNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2) # [Healthy, Tumor]
        )

    def forward(self, x):
        return self.network(x)

class PatientAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.model = DiagnosticNet()
        # In a real project, you'd load pre-trained weights here
        # self.model.load_state_dict(torch.load('weights.pth'))
        
        self.model.eval()
        # avoiding droput during deployment: so that the model wont dropout neurons
        # batch normalization: calculate average output for every neuron
        # active during prediction not training

    def get_strategic_profile(self):
        """Uses the NN to identify only the cancer cells and calculate their strength."""
        # We only care about the cells the NN thinks are 'Tumor' (Index 1)

        # getting the values (not the column names)
        features = self.df[['Gene_A_Oncogene', 'Gene_B_Immune', 'Gene_C_Stromal', 'Gene_D_Therapy', 'Pathway_Score_Inflam']].values
        # assuming the data has been trained, this only looks at the features -> from these factors, 
        # all of the rows will be predicted either tumor (1) or healthy (0) 

        # common scalling all the values -> turning it into torch FloatTensor to reduce memory + gain speed
        X_scaled = torch.FloatTensor(self.scaler.fit_transform(features))
        
        # avoiding back propogation during predicting
        with torch.no_grad():
            # during predicting = giving results
            outputs = self.model(X_scaled) 
            _, predicted = torch.max(outputs, 1) # ex: [1,1,0,1,...] length adjust the size of X_scaled


        # Filter the original dataframe based on the NN's predictions
        malignant_indices = (predicted == 1).numpy() 
        # [True,True,False,True,...] turns into numpy array 
        # True means tumor, false means healthy

        cancer_subset = self.df.iloc[malignant_indices] 
        # only keeps the true values, making it into a subset, 
        # mergining it with the self.df copy, while keeping the features variable the same

        # from the cancer subset, it looks at the max resistance, mean resistance
        return {
            "avg_growth": cancer_subset['Gene_A_Oncogene'].mean() if not cancer_subset.empty else 14.0,
            "max_res_a": cancer_subset['Gene_D_Therapy'].max() if not cancer_subset.empty else 15.0,
            "starting_res_a": cancer_subset['Gene_D_Therapy'].mean() if not cancer_subset.empty else 9.0
        }