import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Construct the path to your final extracted features
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "final", "trip_features.parquet")
    output_dir = os.path.join(base_dir, "outputs", "visualizations")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Dataset not found at: {data_path}")
        return

    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} features.")

    # Drop non-numeric for correlation and plots
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns
    
    sns.set_theme(style="whitegrid")

    # 1. Feature Correlation Matrix
    print("Generating Feature Correlation Matrix...")
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("D-Nerve Routing Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close()

    # Determine Target Variable
    target_col = None
    if 'trip_duration_seconds' in df.columns:
        target_col = 'trip_duration_seconds'
    elif 'actual_duration_sec' in df.columns:
        target_col = 'actual_duration_sec'

    if target_col:
        # 2. Distribution of the Target Variable
        print(f"Generating Distribution Plot for {target_col}...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[target_col], bins=50, kde=True, color='purple')
        plt.title(f"Distribution of {target_col}", fontsize=16)
        plt.xlabel("Seconds")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
        plt.close()

        # 3. Scatter Plot of Distance vs Duration
        distance_col = 'distance_km' if 'distance_km' in df.columns else ('trip_distance_km' if 'trip_distance_km' in df.columns else None)
        if distance_col:
            print("Generating Distance vs Duration Scatter Plot...")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=distance_col, y=target_col, data=df, alpha=0.5, color='teal')
            plt.title(f"Trip Distance vs Actual Duration", fontsize=16)
            plt.xlabel("Distance (km)")
            plt.ylabel("Duration (Seconds)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "distance_vs_duration.png"), dpi=150)
            plt.close()

        # 4. Boxplot by Hour of Day
        time_col = 'hour_of_day' if 'hour_of_day' in df.columns else ('hour' if 'hour' in df.columns else None)
        if time_col:
            print("Generating Duration by Hour Boxplots...")
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=time_col, y=target_col, data=df, palette="viridis")
            plt.title("Trip Duration Patterns by Hour of Day (Traffic Impact)", fontsize=16)
            plt.xlabel("Hour of Day")
            plt.ylabel("Duration (Seconds)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "duration_by_hour.png"), dpi=150)
            plt.close()

    print(f"\n✅ All visualizations generated and saved to: {output_dir}")

if __name__ == "__main__":
    main()
