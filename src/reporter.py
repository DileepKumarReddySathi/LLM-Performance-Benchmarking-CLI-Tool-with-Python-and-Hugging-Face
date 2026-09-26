import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict

class Reporter:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_report(self, results: List[Dict]):
        if not results:
            print("No results to report.")
            return

        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved raw results to {csv_path}")

        # Aggregate metrics per model
        summary = df.groupby("Model_Name").agg(
            Avg_Latency_Sec=("Latency_Seconds", "mean"),
            Avg_Throughput_TPS=("Throughput_TPS", "mean"),
            Peak_RAM_MB=("Peak_Memory_RAM_MB", "max"),
            Peak_VRAM_MB=("Peak_Memory_VRAM_MB", "max"),
            Avg_TTR=("Quality_Metric_Score", "mean")
        ).reset_index()

        # Generate Latency Chart
        plt.figure(figsize=(10, 6))
        plt.bar(summary["Model_Name"], summary["Avg_Latency_Sec"], color='skyblue')
        plt.title('Average Latency per Model')
        plt.xlabel('Model')
        plt.ylabel('Average Latency (Seconds)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        latency_chart_path = os.path.join(self.output_dir, "latency_comparison.png")
        plt.savefig(latency_chart_path)
        plt.close()

        # Generate Memory Chart
        import numpy as np
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(summary["Model_Name"]))
        width = 0.35
        
        plt.bar(x - width/2, summary["Peak_RAM_MB"], width, color='lightgreen', label='RAM')
        if summary["Peak_VRAM_MB"].max() > 0:
            plt.bar(x + width/2, summary["Peak_VRAM_MB"], width, color='orange', label='VRAM')
        
        plt.title('Peak Memory Usage per Model')
        plt.xlabel('Model')
        plt.ylabel('Memory (MB)')
        plt.xticks(x, summary["Model_Name"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        memory_chart_path = os.path.join(self.output_dir, "memory_comparison.png")
        plt.savefig(memory_chart_path)
        plt.close()

        print(f"Saved charts to {self.output_dir}")

        # Print console summary
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(summary.to_string(index=False))
        print("="*50 + "\n")
