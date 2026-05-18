import os
import shutil

# 1. Define your categorized mapping
categories = {
    "Luminance_Test": ["6", "19", "20", "30"],
    "Geometry_Test": ["9", "24", "25", "26"],
    "Subject_Test": ["4", "7", "27", "31"],
    "Details_Test": ["3", "18", "22"]
}

# 2. Set your paths
# Based on your Finder screenshot: Desktop/realfill/RealBench
# Updated paths for safety
base_path = os.path.abspath("./RealBench")
output_path = os.path.abspath("./Qualitative")

def organize_bench():
    # Create the Qualitative directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created base directory: {output_path}")

    for category, samples in categories.items():
        # Create category subfolder (e.g., Qualitative/Luminance_Test)
        cat_dir = os.path.join(output_path, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for sample_id in samples:
            src = os.path.join(base_path, sample_id)
            dst = os.path.join(cat_dir, sample_id)
            
            if os.path.exists(src):
                # We use copytree so the original RealBench remains untouched
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    print(f"✅ Copied {sample_id} to {category}")
                else:
                    print(f"⏩ {sample_id} already exists in {category}, skipping.")
            else:
                print(f"❌ Sample {sample_id} not found in {base_path}")

if __name__ == "__main__":
    organize_bench()