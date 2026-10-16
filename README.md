# SupportPilot

SupportPilot is a local Streamlit demo for AI-assisted support triage. It indexes the fictional Acme Cloud help-center document, retrieves evidence, and drafts a response for a human to review. It never sends replies, issues refunds, or changes customer accounts.

## Setup on macOS (Apple Silicon or Intel)

Use Python 3.11. The project-local environment keeps its Python and pip isolated from Conda and other Python installs:

```bash
cd ~/Downloads/supportpilot
./.venv/bin/python --version
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
```

If `.venv` does not exist yet, create it with Python 3.11 and then run the install and launch commands above:

```bash
python3.11 -m venv .venv
# If python3.11 is not on PATH but installed with Homebrew:
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
```

Do not use a Conda Python to create `.venv`. No activation is required when using the commands above; this avoids mixing an activated Conda base with the virtual environment.

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` for document embeddings and `Qwen/Qwen2.5-1.5B-Instruct` for generation. Hugging Face access and several GB of free disk space may be needed. The included LoRA adapter is loaded by default from `models/supportpilot-lora`; turn off the adapter toggle to run the base model alone. Set `BASE_MODEL` only to the base model that the adapter was trained against, and `ADAPTER_PATH` if the adapter directory is elsewhere.

Automatic file watching is disabled in `.streamlit/config.toml` because Streamlit's module watcher inspects `torch.classes` and can destabilize startup on this setup. Restart the app manually after changing code.

On macOS, embedding inference defaults to one PyTorch CPU thread to avoid a reproducible segmentation fault in the embedding path. Set `SUPPORTPILOT_TORCH_THREADS` to a larger positive integer if you want to try more CPU parallelism on your machine.

## Demo behavior

- The bundled, fictional document is enabled by default. PDF, TXT, and Markdown help documents can also be uploaded.
- Markdown headings such as FAQ questions remain separate chunks. Retrieval uses normalized embeddings and cosine similarity; passages below `MIN_SCORE` (default `0.30`) are excluded, and the model receives only the highest ranked qualifying passage.
- If no passage meets the threshold, the app uses a fixed escalation draft instead of asking the model to invent policy.
- Billing classifications and security-sensitive requests are marked for human review. All drafts require human review before sending.
- Feedback is appended to the ignored local file `data/feedback.jsonl`.

The adapter weights are present in this checkout. The base Qwen weights and embedding model are downloaded on first use. If a model or adapter cannot load, the app displays the underlying setup issue.

## Dataset and evaluation utilities

```bash
./.venv/bin/python -m scripts.make_dataset
./.venv/bin/python -m scripts.validate_dataset data/train.jsonl data/eval.jsonl
./.venv/bin/python -m scripts.run_eval --name baseline --baseline
./.venv/bin/python -m scripts.run_eval --name finetuned
./.venv/bin/python -m scripts.compare
```

Training is documented in `notebooks/finetune_support.ipynb` and is intended for a suitable GPU environment, not a standard Mac laptop.
