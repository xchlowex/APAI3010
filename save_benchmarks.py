def save_to_live_benchmarks(category, sample_name, seed_index, results_dict, filename="live_benchmarks.csv"):
    row = {
        "category": category, 
        "sample": sample_name, 
        "seed": seed_index
    }
    row.update(results_dict)
    
    # Updated fieldnames to include clip_id and dino_id
    fieldnames = ["category", "sample", "seed", "psnr", "ssim", "lpips", "dreamsim", "clip_id", "dino_id"]
    
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        # extrasaction='ignore' ensures it won't crash if a metric is missing
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno()) # This forces the OS to write to the physical SSD NOW