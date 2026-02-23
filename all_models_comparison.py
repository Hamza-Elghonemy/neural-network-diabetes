import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.patches as mpatches

def load_data(filepath='trials_results.csv'):
    """Load the trial results."""
    try:
        df = pd.read_csv(filepath)
        # Sort by F1-Score to have a clear ranking
        return df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

def set_style():
    """Set aesthetic parameters for plots."""
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "#ffffff"})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

def plot_performance_metrics(df, save_dir='results'):
    """Plot Accuracy, Precision, Recall, and F1-Score for all models."""
    plt.figure(figsize=(16, 10))
    
    # Melt dataframe for seaborn barplot
    metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_melted = df.melt(id_vars=['Trial Name'], value_vars=metrics, 
                        var_name='Metric', value_name='Score')
    
    ax = sns.barplot(x='Trial Name', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title('Overall Performance Metrics by Model', pad=20, weight='bold')
    plt.xlabel('Model (Trial Name)', weight='bold')
    plt.ylabel('Score', weight='bold')
    plt.ylim(0.4, 1.0) # Zooming in on the relevant range
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9, rotation=90)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_complexity_vs_performance(df, save_dir='results'):
    """Scatter plot showing Total Parameters vs. F1-Score."""
    plt.figure(figsize=(12, 8))

    sns.scatterplot(data=df, x='Total Parameters', y='F1-Score', hue='Trial Name', 
                    s=400, palette='tab10', alpha=0.8, edgecolor='black')
    
    # Annotate points
    for i in range(len(df)):
        plt.text(df['Total Parameters'].iloc[i] + 20, 
                 df['F1-Score'].iloc[i], 
                 df['Trial Name'].iloc[i], 
                 horizontalalignment='left', 
                 size='medium', color='black', weight='semibold')
                 
    plt.title('Model Complexity vs. Performance (F1-Score)', pad=20, weight='bold')
    plt.xlabel('Total Parameters', weight='bold')
    plt.ylabel('F1-Score', weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Clean up legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complexity_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_efficiency(df, save_dir='results'):
    """Plot Epochs Trained vs Validation Loss."""
    plt.figure(figsize=(12, 8))
    
    # Filter out models with 0 epochs recorded (e.g. tuner models if not logged)
    df_valid = df[df['Epochs Trained'] > 0]
    
    if len(df_valid) == 0:
        print("No valid epoch data for training efficiency plot.")
        return

    sns.scatterplot(data=df_valid, x='Epochs Trained', y='Val Loss', 
                    hue='Trial Name', s=200, palette='Set2', style='Data Strategy', markers=True)
    
    for i in range(len(df_valid)):
        plt.text(df_valid['Epochs Trained'].iloc[i] + 1, 
                 df_valid['Val Loss'].iloc[i], 
                 df_valid['Trial Name'].iloc[i], 
                 horizontalalignment='left', 
                 size='medium', color='black')
                 
    plt.title('Training Efficiency: Epochs vs. Validation Loss', pad=20, weight='bold')
    plt.xlabel('Epochs Trained', weight='bold')
    plt.ylabel('Validation Loss', weight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_insights(df):
    """Print insights comparing the models."""
    print("="*60)
    print("🧠 MODEL COMPARISON INSIGHTS 🧠")
    print("="*60)
    
    # 1. Best Overall Model
    best_f1_idx = df['F1-Score'].idxmax()
    best_model = df.iloc[best_f1_idx]
    
    print("\n🏆 BEST OVERALL MODEL:")
    print(f"  Name: {best_model['Trial Name']}")
    print(f"  F1-Score: {best_model['F1-Score']:.4f}")
    print(f"  Accuracy: {best_model['Test Accuracy']:.4f}")
    print(f"  Strategy: {best_model['Data Strategy']}")
    print(f"  Why it's best: Achieved the highest balance between Precision and Recall (F1-Score), "
          f"making it the most reliable model for predicting both positive and negative diabetes cases.")
          
    # 2. Most Efficient Model
    df['Efficiency_Score'] = df['F1-Score'] / np.log1p(df['Total Parameters'])
    best_efficient = df.loc[df['Efficiency_Score'].idxmax()]
    
    print("\n⚡ MOST EFFICIENT MODEL (Performance per Parameter):")
    print(f"  Name: {best_efficient['Trial Name']}")
    print(f"  F1-Score: {best_efficient['F1-Score']:.4f}")
    print(f"  Parameters: {best_efficient['Total Parameters']}")
    print(f"  Why it's notable: This model punches above its weight. It achieves strong performance "
          f"with a very lightweight architecture, reducing inference time and overfitting risks.")

    # 3. Best Recall (Crucial for Medical Diagnosis)
    best_recall_idx = df['Recall'].idxmax()
    best_recall_model = df.iloc[best_recall_idx]
    
    print("\n🚑 BEST MODEL FOR SCREENING (Highest Recall):")
    print(f"  Name: {best_recall_model['Trial Name']}")
    print(f"  Recall: {best_recall_model['Recall']:.4f}")
    print(f"  F1-Score: {best_recall_model['F1-Score']:.4f}")
    print(f"  Why it matters: In medical settings, failing to identify a diabetic patient (False Negative) "
          f"is often more dangerous than a False Positive. This model minimizes False Negatives best.")
          
    # 4. Impact of Data Strategy
    print("\n📊 DATA STRATEGY ANALYSIS:")
    strategies = df.groupby('Data Strategy')['F1-Score'].mean()
    for strat, mean_f1 in strategies.items():
        print(f"  - {strat}: Average F1-Score = {mean_f1:.4f}")
    
    best_strat = strategies.idxmax()
    print(f"  Conclusion: The '{best_strat}' strategy generally yields the best F1-Scores across architectures.")
    
    print("\n" + "="*60)
    print("Visualizations have been saved to the 'results/' directory:")
    print(" - model_performance_comparison.png")
    print(" - complexity_vs_performance.png")
    print(" - training_efficiency.png")
    print("="*60)

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        set_style()
        plot_performance_metrics(df)
        plot_complexity_vs_performance(df)
        plot_training_efficiency(df)
        generate_insights(df)
