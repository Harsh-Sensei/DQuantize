import json, os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_probs_per_iteration(df, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
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
        fig.savefig(os.path.join(output_dir, f"batch_{batch_idx}_selected_prob.png"))
        plt.show()
        plt.close(fig)

        # --- Unselected probability ---
        fig, ax = plt.subplots()
        ax.plot(x_vals, subdf["prob_unselected"])
        add_block_markers_and_labels(ax)
        ax.set_title(f"Batch {batch_idx}: unselected token probability")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"batch_{batch_idx}_unselected_prob.png"))
        plt.show()
        plt.close(fig)

        # --- Difference ---
        fig, ax = plt.subplots()
        ax.plot(x_vals, subdf["prob_diff"])
        add_block_markers_and_labels(ax)
        ax.set_title(f"Batch {batch_idx}: selected - unselected probability diff")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"batch_{batch_idx}_diff_prob.png"))
        plt.show()
        plt.close(fig)


def plot_aggregated_probs(df, output_dir='plots'):
    """
    Plot aggregated statistics (mean and variance) across all batches.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by block and iter to ensure proper ordering
    df = df.sort_values(["block", "iter"]).reset_index(drop=True)
    
    # Group by block and iter to aggregate across batches
    aggregated = df.groupby(["block", "iter"]).agg({
        "prob_selected": ["mean", "std", "var"],
        "prob_unselected": ["mean", "std", "var"],
        "prob_diff": ["mean", "std", "var"]
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in aggregated.columns.values]
    
    # Compute x-axis indices
    x_vals = range(len(aggregated))
    
    # Find start of each block (where block value changes)
    block_starts = []
    block_labels = []
    prev_block = None
    
    for i, row in aggregated.iterrows():
        if row["block"] != prev_block:
            block_starts.append(i)
            block_labels.append(str(int(row["block"])))
            prev_block = row["block"]
    
    # Calculate midpoints of each block for label positioning
    block_midpoints = []
    for i in range(len(block_starts)):
        start = block_starts[i]
        end = block_starts[i + 1] if i + 1 < len(block_starts) else len(aggregated)
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
        max_iter = aggregated['iter'].max() + 1
        ax.set_xlabel(f"Block ID ({max_iter} Iterations each)")
    
    # --- Selected probability (aggregated) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_vals = aggregated["prob_selected_mean"]
    std_vals = aggregated["prob_selected_std"].fillna(0)  # Fill NaN with 0
    
    ax.plot(x_vals, mean_vals, label='Mean', linewidth=2)
    ax.fill_between(x_vals, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, label='±1 Std Dev', color='blue')
    add_block_markers_and_labels(ax)
    ax.set_title("Aggregated: Selected token probability (Mean ± Std Dev across batches)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_selected_prob.png"), dpi=150)
    plt.show()
    plt.close(fig)
    
    # --- Unselected probability (aggregated) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_vals = aggregated["prob_unselected_mean"]
    std_vals = aggregated["prob_unselected_std"].fillna(0)
    
    ax.plot(x_vals, mean_vals, label='Mean', linewidth=2, color='orange')
    ax.fill_between(x_vals, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, label='±1 Std Dev', color='orange')
    add_block_markers_and_labels(ax)
    ax.set_title("Aggregated: Unselected token probability (Mean ± Std Dev across batches)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_unselected_prob.png"), dpi=150)
    plt.show()
    plt.close(fig)
    
    # --- Difference (aggregated) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_vals = aggregated["prob_diff_mean"]
    std_vals = aggregated["prob_diff_std"].fillna(0)
    
    ax.plot(x_vals, mean_vals, label='Mean', linewidth=2, color='green')
    ax.fill_between(x_vals, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, label='±1 Std Dev', color='green')
    add_block_markers_and_labels(ax)
    ax.set_title("Aggregated: Selected - Unselected probability diff (Mean ± Std Dev across batches)")
    ax.set_ylabel("Probability Difference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_diff_prob.png"), dpi=150)
    plt.show()
    plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("AGGREGATED STATISTICS SUMMARY")
    print("="*80)
    print(f"Number of batches: {df['batch_idx'].nunique()}")
    print(f"Number of timesteps: {len(aggregated)}")
    print(f"\nSelected Probability:")
    print(f"  Mean: {aggregated['prob_selected_mean'].mean():.6f} ± {aggregated['prob_selected_mean'].std():.6f}")
    print(f"  Std Dev: {aggregated['prob_selected_std'].mean():.6f} ± {aggregated['prob_selected_std'].std():.6f}")
    print(f"\nUnselected Probability:")
    print(f"  Mean: {aggregated['prob_unselected_mean'].mean():.6f} ± {aggregated['prob_unselected_mean'].std():.6f}")
    print(f"  Std Dev: {aggregated['prob_unselected_std'].mean():.6f} ± {aggregated['prob_unselected_std'].std():.6f}")
    print(f"\nDifference:")
    print(f"  Mean: {aggregated['prob_diff_mean'].mean():.6f} ± {aggregated['prob_diff_mean'].std():.6f}")
    print(f"  Std Dev: {aggregated['prob_diff_std'].mean():.6f} ± {aggregated['prob_diff_std'].std():.6f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze probability distributions from LLaDA generation')
    parser.add_argument('--probs_file', type=str, required=True,
                        help='Path to probs.jsonl file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (default: same directory as probs_file)')
    
    args = parser.parse_args()
    
    # If output_dir not specified, use the same directory as probs_file
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.probs_file) if os.path.dirname(args.probs_file) else 'plots'
    
    # -------------------------------------------------
    # Load JSON Lines into a DataFrame (vectorized)
    # -------------------------------------------------
    df = pd.read_json(args.probs_file, lines=True)

    # Expand probs into two columns, replace -inf with 0
    # If probs has more than two elements, use the last two of a descending sorted array (two smallest values)
    def get_two_smallest(probs_list):
        if len(probs_list) > 2:
            sorted_probs = sorted(probs_list, reverse=True)  # descending order
            return sorted_probs[-2:]  # last two elements (two smallest)
        return probs_list
    
    processed_probs = df["probs"].apply(get_two_smallest).tolist()
    df[["prob_selected", "prob_unselected"]] = pd.DataFrame(processed_probs, index=df.index)
    
    # Assert that only prob_unselected can be -inf
    prob_selected_inf = df["prob_selected"] == -float("inf")
    if prob_selected_inf.any():
        raise AssertionError(f"Found -inf in prob_selected at indices: {df[prob_selected_inf].index.tolist()}. Only prob_unselected can be -inf.")
    
    # If prob_unselected is -inf, set it to the same value as prob_selected
    prob_unselected_inf = df["prob_unselected"] == -float("inf")
    if prob_unselected_inf.any():
        df.loc[prob_unselected_inf, "prob_unselected"] = df.loc[prob_unselected_inf, "prob_selected"]

    # Compute probability difference
    df["prob_diff"] = df["prob_selected"] - df["prob_unselected"]

    # Vectorized ordering of timesteps (block:iter)
    df["timestep"] = df["block"].astype(str) + ":" + df["iter"].astype(str)

    df = df.sort_values(["batch_idx", "block", "iter"]).reset_index(drop=True)

    print(df.head())
    
    # Plot individual batch plots
    plot_probs_per_iteration(df, args.output_dir)
    
    # Plot aggregated statistics across all batches
    if df['batch_idx'].nunique() > 1:
        print("\nGenerating aggregated plots...")
        plot_aggregated_probs(df, args.output_dir)
    else:
        print("\nOnly one batch found. Skipping aggregated plots.")


if __name__ == '__main__':
    main()

