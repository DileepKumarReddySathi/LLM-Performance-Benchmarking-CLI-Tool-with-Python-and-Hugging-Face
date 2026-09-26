# LLM Performance Benchmarking CLI Tool 🚀

A professional Python-based Command-Line Interface (CLI) application designed for Machine Learning Operations (MLOps) teams to systematically benchmark and compare the performance of Large Language Models (LLMs) from the Hugging Face ecosystem.

This tool automates the profiling of multiple models across standardized datasets to help teams make informed decisions balancing generation speed, resource consumption, and text output quality.

## 🌟 Key Capabilities

*   **Dynamic Model Loading:** Evaluates any Causal LM available on the Hugging Face Hub based on a YAML/JSON configuration. No hardcoded logic.
*   **Performance Profiling:** Accurately measures end-to-end inference latency and computes token throughput (Tokens-Per-Second).
*   **Hardware Resource Monitoring:** Tracks peak System RAM utilization (`psutil`) and automatically handles GPU VRAM monitoring (`torch.cuda`) if an NVIDIA GPU is available.
*   **Automated Quality Assurance:** Calculates the Type-Token Ratio (TTR) as a lexical diversity metric to programmatically detect and flag degenerate or infinitely repeating output loops.
*   **Visual Data Reporting:** Aggregates telemetry data to generate comparison bar charts and raw CSV datasets for comprehensive MLOps analysis.
*   **Robust Memory Management:** Implements aggressive garbage collection and CUDA cache clearing between model runs to prevent Out-Of-Memory (OOM) crashes during batch testing.

---

## 🛠️ System Architecture

1.  **Configuration Layer (`config_parser.py`):** Validates and parses user-provided configurations.
2.  **Core Benchmarking Engine (`model_runner.py`):** Encapsulates the `transformers` logic for loading models/tokenizers dynamically and executing inference on CPU or GPU.
3.  **Hardware Profiler (`profiler.py`):** Wraps inference functions to extract precise timings and hardware memory states.
4.  **Reporting Layer (`reporter.py`):** Aggregates telemetry using `pandas` and renders charts via `matplotlib`.

---

## 📦 Installation & Setup

### Option A: Local Python Environment
Recommended for quick tests or running on local hardware. Requires Python 3.10+.

```bash
# 1. Clone or navigate to the repository
cd llm_benchmarker

# 2. Create and activate a virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B: Dockerized Environment
Recommended for cross-platform reproducibility and isolated dependency management.

```bash
# 1. Copy the example environment variables
cp .env.example .env

# 2. (Optional) If you have a GPU, uncomment the `deploy: gpu` block in docker-compose.yml

# 3. Build the container
docker-compose build
```

---

## ⚙️ Configuration Guide

The CLI is driven entirely by a configuration file. Create a `config.yaml` or `config.json`. 

Example `config.yaml`:
```yaml
models:
  - "sshleifer/tiny-gpt2"
  - "hf-internal-testing/tiny-random-gpt2"
  - "sshleifer/tiny-ctrl"
dataset_path: "datasets/sample_prompts.jsonl"
max_new_tokens: 50
```

*   `models`: A list containing at least three valid Hugging Face model IDs.
*   `dataset_path`: Path to a `.jsonl` file where each line contains at least a `"prompt"` key.
*   `max_new_tokens`: Maximum length of the generated sequence.

---

## 🚀 Usage

Invoke the CLI by passing the configuration file.

**Locally:**
```bash
python src/cli.py --config config.yaml
```

**Via Docker:**
```bash
docker-compose run app python src/cli.py --config config.yaml
```

During execution, a progress bar will appear. The system will load each model sequentially, process all prompts in the dataset, clean up system memory, and move to the next model.

---

## 📊 Outputs & Artifacts

Upon completion, the tool will output a formatted console table and save detailed artifacts to the `output/` directory:

1.  `results.csv`: The raw telemetry data including Latency, Throughput (TPS), Peak RAM/VRAM, and TTR for every single prompt/model combination.
2.  `latency_comparison.png`: A bar chart visualizing the Average Latency across all models.
3.  `memory_comparison.png`: A side-by-side bar chart showing Peak RAM and Peak VRAM utilization across models.

---

## 🧠 Metrics Explained
*   **Latency (Seconds):** Total time elapsed from prompt submission to final token generation.
*   **Throughput (TPS):** The generation speed, calculated as `Generated Tokens / Latency`.
*   **Peak Memory (MB):** The maximum amount of Resident Set Size (RSS) RAM or GPU VRAM allocated during the generation phase.
*   **Type-Token Ratio (TTR):** Number of unique words divided by the total number of words. A score closer to 1.0 indicates high diversity, while a score near 0.0 indicates a model stuck in a repetitive loop.
