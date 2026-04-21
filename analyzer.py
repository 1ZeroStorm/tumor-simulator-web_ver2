import pandas as pd

class PatientAnalyzer:
    def __init__(self, csv_path=None, df=None):
        # Imports the CSV as a pandas DataFrame or accepts a DataFrame directly
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either csv_path or df must be provided")

    def get_strategic_profile(self):
        """Filters for cancer cells and calculates their strength based on raw data."""
        
        # Deletes the rows that are healthy by only keeping 'Tumor' status
        cancer_subset = self.df[self.df['Disease_Status'] == 'Tumor']

        # Calculates metrics directly from the filtered dataframe
        return {
            "avg_growth": cancer_subset['Gene_A_Oncogene'].mean() if not cancer_subset.empty else 14.0,
            "max_res_a": cancer_subset['Gene_D_Therapy'].max() if not cancer_subset.empty else 15.0,
            "starting_res_a": cancer_subset['Gene_D_Therapy'].mean() if not cancer_subset.empty else 9.0
        }

    def get_patient_profile(self, data=None):
        """Alias for get_strategic_profile for compatibility."""
        if data is not None:
            self.df = data
        return self.get_strategic_profile()

# Example of how to run it:
if __name__ == "__main__":
    # Point to your synthetic CSV file
    analyzer = PatientAnalyzer("Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv")
    
    # Get the calculated values
    profile = analyzer.get_strategic_profile()
    
    # Print the results
    print("Strategic Profile:")
    for key, value in profile.items():
        print(f"{key}: {value:.4f}")