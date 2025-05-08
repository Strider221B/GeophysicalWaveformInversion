import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from configs.config import Config
from helpers.constants import Constants

class PlotHelper:

    @staticmethod
    def plot_history(history: List[Dict[str, any]]):
        print("\n--- Plotting Training History ---")
        if history:
            try:
                hist_df = pd.DataFrame(history)
                plt.figure(figsize=(12, 6))
                plt.plot(hist_df["epoch"], hist_df["train_loss"], "o-", label="Train Loss")
                # Only plot validation loss if it exists and is not all None/NaN
                if "valid_loss" in hist_df.columns and not hist_df["valid_loss"].isnull().all():
                    plt.plot(
                        hist_df["epoch"],
                        hist_df["valid_loss"],
                        "s--",  # Square markers, dashed line
                        label="Validation Loss",
                    )
                plt.title("Training and Validation Loss vs. Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (Mean Absolute Error)")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.ylim(bottom=0)  # Loss should not be negative
                plt.tight_layout()
                plot_fname = os.path.join(Config.working_dir, Constants.PNG_TRAINING_HISTORY)
                plt.savefig(plot_fname)
                print(f"Saved history plot: {plot_fname}")
                plt.show()  # Display the plot
            except Exception as e:
                print(f"E: Failed plotting training history: {e}")
        else:
            print("No training history recorded to plot.")
