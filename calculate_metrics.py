import pandas as pd
import os

def analyze_benchmarks(filename="live_benchmarks.csv", output_file="final_summary.csv"):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Did the overnight run finish?")
        return

    # 1. Load the data
    df = pd.read_csv(filename)
    
    # 2. Define weights for the 'Success Score' 
    # We weight DINO higher (0.7) because it proves our Geometric Fusion branch works
    w_dino = 0.7
    w_clip = 0.3
    
    # 3. Calculate the Weighted Sum for every seed
    df['weighted_score'] = (df['dino_id'] * w_dino) + (df['clip_id'] * w_clip)

    # 4. Group by Category and Sample to get averages
    summary = df.groupby(['category', 'sample']).agg({
        'clip_id': 'mean',
        'dino_id': 'mean',
        'weighted_score': 'mean',
        'psnr': 'mean',
        'ssim': 'mean',
        'lpips': 'mean'
    }).reset_index()

    # 5. Find the MAX weighted sum (The "Best Seed" performance)
    # This helps show the upper-bound potential of the Geometric Fusion
    best_seeds = df.loc[df.groupby(['category', 'sample'])['weighted_score'].idxmax()]
    best_seeds = best_seeds[['category', 'sample', 'seed', 'weighted_score']]
    best_seeds.columns = ['category', 'sample', 'best_seed_index', 'max_weighted_score']

    # 6. Merge averages with the Max performance data
    final_report = pd.merge(summary, best_seeds, on=['category', 'sample'])

    # 7. Export and Print
    # 7. Export and Print
    # Check if file exists to decide if we need to write the header
    file_exists = os.path.isfile(output_file)
    
    final_report.to_csv(
        output_file, 
        mode='a',              # 'a' stands for append
        index=False, 
        header=not file_exists # Only write header if the file is new
    )
    
    print("\n--- PERFORMANCE SUMMARY BY CATEGORY ---")
    print(final_report[['category', 'sample', 'dino_id', 'max_weighted_score']])
    
    return final_report

if __name__ == "__main__":
    analyze_benchmarks()