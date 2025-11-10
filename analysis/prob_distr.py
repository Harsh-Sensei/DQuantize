import json, os
import pandas as pd
import matplotlib.pyplot as plt


def plot_probs_per_iteration(df):
    os.makedirs('plots', exist_ok=True)
    # Group by batch_idx and plot each metric
    for batch_idx, subdf in df.groupby("batch_idx"):

        subdf = subdf.sort_values(["block", "iter"])
        # Compute x-axis indices (integer positions)
        x_vals = range(len(subdf))

        # Find start of each block (where block value changes)
        block_starts = []
        block_labels = []
        prev_block = None
        
        for i, (idx, row) in enumerate(subdf.iterrows()):
            if row["block"] != prev_block:
                block_starts.append(i)
                block_labels.append(str(row["block"]))
                prev_block = row["block"]
        
        # Calculate midpoints of each block for label positioning
        block_midpoints = []
        for i in range(len(block_starts)):
            start = block_starts[i]
            end = block_starts[i + 1] if i + 1 < len(block_starts) else len(subdf)
            midpoint = (start + end) / 2
            block_midpoints.append(midpoint)

        def add_block_markers_and_labels(ax):
            """Add vertical dashed lines at each new block start and label x-axis with block IDs."""
            # Add vertical lines at block boundaries
            for idx in block_starts[1:]:  # Skip first one (it's at 0)
                ax.axvline(x=idx, linestyle="--", alpha=0.5, color='gray')
            
            # Set x-axis ticks at block midpoints with block labels
            ax.set_xticks(block_midpoints)
            ax.set_xticklabels(block_labels)
            ax.set_xlabel(f"Block ID ({subdf['iter'].max()+1} Iterations each)")

        # --- Selected probability ---
        fig, ax = plt.subplots()
        ax.plot(x_vals, subdf["prob_selected"])
        add_block_markers_and_labels(ax)
        ax.set_title(f"Batch {batch_idx}: selected token probability")
        plt.tight_layout()
        fig.savefig(f"plots/batch_{batch_idx}_selected_prob.png")
        plt.show()
        plt.close(fig)

        # --- Unselected probability ---
        fig, ax = plt.subplots()
        ax.plot(x_vals, subdf["prob_unselected"])
        add_block_markers_and_labels(ax)
        ax.set_title(f"Batch {batch_idx}: unselected token probability")
        plt.tight_layout()
        fig.savefig(f"plots/batch_{batch_idx}_unselected_prob.png")
        plt.show()
        plt.close(fig)

        # --- Difference ---
        fig, ax = plt.subplots()
        ax.plot(x_vals, subdf["prob_diff"])
        add_block_markers_and_labels(ax)
        ax.set_title(f"Batch {batch_idx}: selected - unselected probability diff")
        plt.tight_layout()
        fig.savefig(f"plots/batch_{batch_idx}_diff_prob.png")
        plt.show()
        plt.close(fig)


def main():
    # -------------------------------------------------
    # Load JSON Lines into a DataFrame (vectorized)
    # -------------------------------------------------
    LOG_FILE = "logs/probs.jsonl"  # <-- change this
    df = pd.read_json(LOG_FILE, lines=True)

    # Expand probs into two columns, replace -inf with 0
    df[["prob_selected", "prob_unselected"]] = pd.DataFrame(df["probs"].tolist(), index=df.index)
    df["prob_selected"] = df["prob_selected"].replace([-float("inf")], 0.0)
    df["prob_unselected"] = df["prob_unselected"].replace([-float("inf")], 0.0)

    # Compute probability difference
    df["prob_diff"] = df["prob_selected"] - df["prob_unselected"]

    # Vectorized ordering of timesteps (block:iter)
    df["timestep"] = df["block"].astype(str) + ":" + df["iter"].astype(str)

    df = df.sort_values(["batch_idx", "block", "iter"]).reset_index(drop=True)

    print(df.head())
    plot_probs_per_iteration(df)


if __name__ == '__main__':
    main()

