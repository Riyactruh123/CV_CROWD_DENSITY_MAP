import os
import h5py
import csv
import numpy as np

def get_predicted_counts_from_h5(folder_path):
    """
    Loads .h5 files from a folder, computes predicted count (sum of density map).
    """
    results = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)

            with h5py.File(file_path, 'r') as f:
                # Try common dataset key names
                if 'density' in f:
                    density_map = np.array(f['density'])
                elif 'pred' in f:
                    density_map = np.array(f['pred'])
                else:
                    print(f"⚠️ No recognized dataset in {filename}. Skipping.")
                    continue

                predicted_count = np.sum(density_map)
                results.append((filename, predicted_count))
                print(f"{filename}: {predicted_count:.2f}")

    return results

def save_to_csv(results, output_csv):
    """
    Saves predicted counts to a CSV.
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'predicted_count'])
        for fname, count in results:
            writer.writerow([fname, f"{count:.2f}"])

def main():
    input_folder = "/Users/kushireddy/SINet/dataset/ShanghaiTech/part_A/train_data/density_map"               # Folder with .h5 predicted density maps
    output_csv = "predicted_counts.csv"         # Output file name

    results = get_predicted_counts_from_h5(input_folder)
    save_to_csv(results, output_csv)

    print(f"\n✅ Saved predicted counts to: {output_csv}")

if __name__ == "__main__":
    main()
