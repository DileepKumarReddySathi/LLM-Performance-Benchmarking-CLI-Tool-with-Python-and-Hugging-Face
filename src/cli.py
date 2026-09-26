import argparse
import json
import logging
from tqdm import tqdm
import sys
import os

# Add the src directory to the path so we can import the other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_parser import load_config
from model_runner import HuggingFaceRunner
from profiler import profile_inference
from reporter import Reporter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_dataset(dataset_path: str):
    prompts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.jsonl'):
            for line in f:
                data = json.loads(line)
                prompts.append((data.get('id', 'unknown'), data['prompt']))
        else:
            raise ValueError("Unsupported dataset format. Only .jsonl is supported for now.")
    return prompts

def main():
    parser = argparse.ArgumentParser(description="LLM Performance Benchmarking CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (YAML/JSON).")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        prompts = load_dataset(config.dataset_path)
    except Exception as e:
        logging.error(f"Initialization Error: {e}")
        return

    all_results = []
    reporter = Reporter()

    for model_id in config.models:
        logging.info(f"Starting benchmark for model: {model_id}")
        try:
            runner = HuggingFaceRunner(model_id)
        except Exception as e:
            logging.error(f"Failed to load model {model_id}: {e}")
            continue

        for prompt_id, prompt_text in tqdm(prompts, desc=f"Benchmarking {model_id}"):
            try:
                metrics = profile_inference(runner.generate, prompt_text, config.max_new_tokens)
                
                result = {
                    "Model_Name": model_id,
                    "Prompt_ID": prompt_id,
                    "Prompt_Text": prompt_text,
                    "Latency_Seconds": metrics["latency_sec"],
                    "Throughput_TPS": metrics["throughput_tps"],
                    "Peak_Memory_RAM_MB": metrics["peak_ram_bytes"] / (1024 * 1024),
                    "Peak_Memory_VRAM_MB": metrics["peak_vram_bytes"] / (1024 * 1024),
                    "Quality_Metric_Score": metrics["quality_metric_ttr"],
                }
                all_results.append(result)
            except Exception as e:
                logging.error(f"Error during inference for model {model_id} on prompt {prompt_id}: {e}")
                
        # Clean up model
        runner.cleanup()

    reporter.generate_report(all_results)

if __name__ == "__main__":
    main()
