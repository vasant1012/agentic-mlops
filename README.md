# MLflow Experiment Analysis Agent 🧑‍🔬🤖

A modular **LangChain + Hugging Face** project that leverages **Qwen2-1.5B-Instruct** to analyze MLflow experiments.  
This agent can:
- Load experiment results from JSON
- Query MLflow tracking servers directly
- Generate structured experiment insights using LLMs

---

## 📂 Project Structure

```
.
mlops_agents/
│── __init__.py
│── main.py                        # Entry point (choose agent and run)
│
├── utils/
│   ├── __init__.py
│   ├── mlflow_utils.py            # Functions: get_runs_summary, format_runs_for_llm
│   └── data_models.py             # RunSummary dataclass
│   └── logger.py                  # custom logger 
│
├── tools/
│   ├── __init__.py
│   ├── mlflow_tools.py            # LangChain tools: load_mlflow_experiments_from_json query_mlflow_experiment
│   └── analysis_tools.py          # analyze_experiment_performance
│
├── agents/
│   ├── __init__.py
│   ├── langchain_agent.py         # MLflowAnalysisAgent (LangChain ReAct Agent)
│   └── direct_analyzer.py         # DirectTransformerMLflowAnalyzer
│
└── prompts/
│   ├── __init__.py
│   └── system_prompt.py           # SYSTEM_PROMPT definition
│
├── ml_logs_pipeline.py           # main driver code
├── log_analyzer_agent.py          # reference agent pipeline
├── train_with_mlflow.py          # mlflow pipeline to generate logs
├── README.md                  # Project documentation
└── .gitignore
````

---
## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/your-username/agentic-mlops.git
cd agentic-mlops
````

Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

Main libraries used:

* [LangChain](https://python.langchain.com/)
* [Transformers](https://huggingface.co/docs/transformers/index)
* [Torch](https://pytorch.org/)
* [MLflow](https://mlflow.org/)
* [HuggingFace Hub](https://huggingface.co/)

---

## 🚀 Usage

1. **Run the main script**
   From the project root, execute:

   ```bash
   python ml_logs_pipeline.py
   ```

2. **Experiment log check**

   * If a file named `mlflow_experiments.json` already exists, it will be used directly.
   * If not, the script will:

     * Fetch MLflow runs from the experiment `iris_demo` (latest 5 runs).
     * Save the results into a JSON file (`mlflow_experiments.json`).

3. **Choose analysis mode**
   After the MLflow logs are prepared, you will be prompted with:

   ```
   Choose implementation:
   1. LangChain Agent
   2. Direct Transformer
   ```

   * Enter **1** → Uses a **LangChain Agent** (`agents/langchain_agent.py`) powered by `Qwen/Qwen2-1.5B-Instruct` to analyze MLflow runs.
   * Enter **2** → Uses a **Direct Transformer Analyzer** (`agents/direct_analyzer.py`) to analyze the MLflow logs directly.

4. **View results**
   The selected analyzer will output insights about the MLflow experiment runs in your terminal.

---

Would you like me to also add **example CLI input/output** (like a mock run showing logs + choice + model output) so the usage feels more hands-on?

---

## 📂 Example JSON Format

`examples/mlflow_experiments.json`

```json
{
  "run_0": {
    "run_id": "abc123",
    "params": {"learning_rate": "0.01", "batch_size": "32"},
    "metrics": {"accuracy": 0.91, "loss": 0.12}
  },
  "run_1": {
    "run_id": "def456",
    "params": {"learning_rate": "0.001", "batch_size": "64"},
    "metrics": {"accuracy": 0.94, "loss": 0.09}
  }
}
```

---

## 🛠 Tools

* **`load_mlflow_experiments_from_json`** → Load experiments from JSON file
* **`query_mlflow_experiment`** → Query MLflow tracking server by experiment name
* **`analyze_experiment_performance`** → Generate structured analysis prompt for the LLM

---

## 🔬 Example Queries

1. Load from JSON:

```python
agent.analyze_from_json("examples/mlflow_experiments.json")
```

2. Query MLflow server:

```python
agent.analyze_from_mlflow("my_experiment", top_n=10)
```

3. Custom query:

```python
agent.custom_analysis("What trends do you see in validation accuracy?")
```

---

## ⚡️ Roadmap

* [ ] Add **quantization support** for Qwen2 models (BitsAndBytesConfig)
* [ ] Enable **multi-experiment comparison**
* [ ] Add **Streamlit dashboard** for interactive analysis
* [ ] Extend support to **RAG-style context enrichment**

---

## 📜 License

MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Contributions are welcome!
Please fork the repo and submit a PR with enhancements or bug fixes.

---

## 🙌 Acknowledgements

* [Qwen Team](https://huggingface.co/Qwen) for Qwen2 models
* [LangChain](https://www.langchain.com/) for agent framework
* [MLflow](https://mlflow.org/) for experiment tracking
