import numpy as np
import os
import time

import matplotlib.pyplot as plt

def load_data(file_path):
    return np.load(file_path)

def plot_data(data, ax, title):
    ax.clear()
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_title(title)

def main():
    file_paths = ["plot_points/" + file for file in os.listdir("plot_points") if file.endswith(".npy")]
    print(f"Found {len(file_paths)} files")
    fig, axs = plt.subplots(len(file_paths), 1, figsize=(10, 8))
    plt.ion()
    
    if len(file_paths) == 1:
        axs = [axs]
    
    previous_data = [None] * len(file_paths)
        
    step = 0

    while True:
        for i, file_path in enumerate(file_paths):
            if os.path.exists(file_path):
                data = load_data(file_path)
                if previous_data[i] is None or not np.array_equal(data, previous_data[i]):
                    plot_data(data, axs[i], f"Plot {file_path.split('.')[0]}")
                    previous_data[i] = data
        
        plt.draw()
        plt.pause(0.1)
        
        step += 1
        if step % 100 == 0:
            print(f"Step {step}: Updated plots")
                
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()