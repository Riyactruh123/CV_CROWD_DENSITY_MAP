import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv

def load_density_maps_h5(folder):
    """
    Load all .h5 density maps from the given folder.
    """
    maps = []
    filenames = []

    for file in sorted(os.listdir(folder)):
        if file.endswith('.h5'):
            path = os.path.join(folder, file)
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    density_map = np.array(f[key])
                    if density_map.ndim >= 2:
                        maps.append(density_map)
                        filenames.append(file)
                        break
    return maps, filenames

def load_ground_truth(csv_file):
    """
    Load ground truth counts from CSV.
    CSV format: filename,count
    """
    gt_counts = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            filename, count = row
            gt_counts[filename] = float(count)
    return gt_counts

def compute_metrics(pred_maps, filenames, gt_counts):
    """
    Compute MAE and RMSE between predicted and actual counts.
    """
    pred_counts = [np.sum(dm) for dm in pred_maps]
    errors = []
    results = []

    for fname, pcount in zip(filenames, pred_counts):
        if fname not in gt_counts:
            print(f"⚠️ Ground truth missing for {fname}")
            continue
        gcount = gt_counts[fname]
        error = abs(pcount - gcount)
        errors.append(error)
        results.append((fname, pcount, gcount, error))

    if errors:
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.square(errors)))
    else:
        mae = rmse = float('nan')

    return mae, rmse, results

def save_results_to_csv(results, output_file):
    """
    Save prediction results to a CSV file.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'predicted_count', 'ground_truth_count', 'error'])
        for row in results:
            writer.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}"])

def show_density_map(density_map, title="Density Map"):
    """
    Show a density map with matplotlib.
    """
    plt.imshow(density_map, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def main():
    pred_folder = "/Users/kushireddy/SINet/dataset/ShanghaiTech/part_A/train_data/density_map"                  # Folder with .h5 predicted maps
    gt_csv = "/Users/kushireddy/SINet/predicted_counts.csv"             # CSV with filename,count
    output_csv = "prediction_results.csv"          # Output file to save results

    pred_maps, filenames = load_density_maps_h5(pred_folder)
    gt_counts = load_ground_truth(gt_csv)

    mae, rmse, results = compute_metrics(pred_maps, filenames, gt_counts)

    print("=== Prediction Error Summary ===")
    for fname, p, g, e in results:
        print(f"{fname} | Predicted: {p:.2f} | Ground Truth: {g:.2f} | Error: {e:.2f}")

    print(f"\n🔍 Overall MAE: {mae:.2f}")
    print(f"📈 Overall RMSE: {rmse:.2f}")

    save_results_to_csv(results, output_csv)
    print(f"\n📁 Results saved to: {output_csv}")

    # Optional: show first few maps
    for i in range(min(3, len(pred_maps))):
        title = f"{filenames[i]} - Predicted: {np.sum(pred_maps[i]):.2f}"
        show_density_map(pred_maps[i], title=title)

if __name__ == "__main__":
    main()
