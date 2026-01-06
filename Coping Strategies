import os
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_anxiety_data():
    # Extract the newly uploaded archive
    dataset_folder = "anxiety_depression_data.csv"
    
    # List files to verify the exact CSV name
    print("Files in dataset:", os.listdir(dataset_folder))
    
    # Path to the CSV file
    data_path = os.path.join(dataset_folder, 'anxiety_depression_data.csv')
    
    # Load the CSV file
    df = pd.read_csv(data_path)
    # Create binary Anxiety Group based on median score
    df['Anxiety_Group'] = (df['Anxiety_Score'] > df['Anxiety_Score'].median()).astype(int)

    # Create a single figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
# Plot 1: Therapy vs. Anxiety Group
    sns.countplot(data=df, x='Therapy', hue='Anxiety_Group', ax=ax1)
    ax1.set_title('Therapy Usage vs. Anxiety Group')
    ax1.set_xlabel('Therapy (0 = No, 1 = Yes)')
    ax1.set_ylabel('Count')

    # Plot 2: Meditation vs. Anxiety Group
    sns.countplot(data=df, x='Meditation', hue='Anxiety_Group', ax=ax2)
    ax2.set_title('Meditation Usage vs. Anxiety Group')
    ax2.set_xlabel('Meditation (0 = No, 1 = Yes)')
    ax2.set_ylabel('Count')

    # Plot 3: Substance Use vs. Anxiety Group
    sns.countplot(data=df, x='Substance_Use', hue='Anxiety_Group', ax=ax3)
    ax3.set_title('Substance Use vs. Anxiety Group')
    ax3.set_xlabel('Substance Use Category')
    ax3.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_anxiety_data()
