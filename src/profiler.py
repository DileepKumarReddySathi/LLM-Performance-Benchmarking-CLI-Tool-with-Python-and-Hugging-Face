import time
import psutil
import os
import torch
import re

def calculate_ttr(text: str) -> float:
    """Calculates the Type-Token Ratio (TTR) for the given text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def profile_inference(inference_func, prompt: str, max_new_tokens: int):
    """
    Profiles the execution of an inference function.
    Returns: Dictionary containing metrics.
    """
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.perf_counter()
    output_text, token_count = inference_func(prompt, max_new_tokens)
    end_time = time.perf_counter()
    
    end_ram = process.memory_info().rss
    latency = end_time - start_time
    throughput = token_count / latency if latency > 0 else 0
    ttr = calculate_ttr(output_text)
    
    metrics = {
        "latency_sec": latency,
        "throughput_tps": throughput,
        "peak_ram_bytes": max(start_ram, end_ram),
        "quality_metric_ttr": ttr,
        "generated_text": output_text
    }
    
    if torch.cuda.is_available():
        metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    else:
        metrics["peak_vram_bytes"] = 0
        
    return metrics
