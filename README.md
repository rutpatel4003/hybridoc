# ⚡ Private-RAG: Extreme-Privacy Document Intelligence

![Project Header](https://img.shields.io/badge/Status-Beta-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Engine-Ollama%20Qwen3-orange)
![GPU](https://img.shields.io/badge/Optimized-RTX%203060-green)

> "Because your private data shouldn't be the price for AI intelligence."

**Private-RAG** is a localized Retrieval-Augmented Generation pipeline built for developers who care about data sovereignty. Forget sending PDFs to a cloud API—this project runs a full-scale search and reasoning engine entirely on your own silicon.

---

## 🛠 Why This Matters (The Engineering Edge)
Most RAG "tutorials" are basic. This project implements advanced retrieval techniques used in production:
- **Hybrid Retrieval Engine**: Combines **BM25** (keyword frequency) with **Vector Embeddings** (semantic meaning). It finds the "exact needle" even when the "haystack" is a complex technical PDF.
- **FlashRank Re-ranking**: We don't just grab the top 10 results; we use a secondary, specialized cross-encoder model to re-score them, ensuring the LLM only sees the most high-signal context.
- **Graph-Based State Machine**: Built with **LangGraph**, the logic is structured as a controlled workflow rather than a messy script, allowing for predictable and debuggable AI behavior.

---

## ⚙️ Configuration & Hardware Tuning

Performance on an RTX 3060 depends on how you tune the `config.py`. Here is how to master the settings:

### **The "Speed vs. Accuracy" Trade-off**
In `config.py`, you can modify these variables to suit your needs:

| Variable | Recommendation | Impact |
| :--- | :--- | :--- |
| `CHUNK_SIZE` | **512** | **Smaller is better** for technical exams or code snippets. **1024** is better for general prose. | Default: 1024
| `CONTEXTUALIZE_CHUNKS` | **False** (Default) | Turn this to **True** if your document chunks feel "lost." It uses an LLM to explain each chunk before indexing. *Note: Increases ingestion time.* | Default: True
| `N_CONTEXT_RESULTS` | **3 to 5** | How many sources the LLM sees. Too many can cause "Lost in the Middle" syndrome where the AI gets confused. |

---

## 🚀 Installation & "Run-In-60-Seconds"

### 1. The Brain (Ollama)
Ensure Ollama is running and you have the model:
```bash
ollama pull qwen3:4b
```

### 2. The Setup 
```bash
# Clone the vision
git clone [https://github.com/YOUR_USERNAME/private-rag.git](https://github.com/YOUR_USERNAME/private-rag.git)
cd private-rag

# Install dependencies
pip install -r requirements.txt
```

### 3. The Execution
```bash
streamlit run app.py
```
