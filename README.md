# ⚡ LLM Tokenizer & Cost Intelligence Platform

An end-to-end NLP engineering tool and cost intelligence dashboard. This platform combines an algorithmic **Byte-Pair Encoding (BPE) tokenizer built from scratch** with production benchmarks (`tiktoken`) to deliver real-time token boundary visualization, multi-model cost analytics, and prompt optimization.

---

## 🚀 Key Features

* **Custom BPE Engine from Scratch:** Pure Python implementation covering regex pre-tokenization (`\p{L}`, `\p{N}`), UTF-8 byte handling, iterative pair-frequency counting, and vocabulary serialization.
* **Production Model Benchmarking:** Compares subword token counts across OpenAI (GPT-4o, GPT-4o-mini, GPT-4 Turbo), Anthropic (Claude 3.5 Sonnet), and Google (Gemini 1.5 Flash).
* **Token Boundary Visualizer:** Highlights alternating subword segments with whitespace tracking (`␣`) and hover tooltips for raw token IDs and UTF-8 byte arrays.
* **Prompt Optimization Engine:** Automatically detects conversational filler and redundant formatting to propose compressed prompts, estimating percentage savings and cost reductions per 1M calls.
* **Full-Stack Architecture:** FastAPI backend with Pydantic validation connected to a React, TypeScript, and Tailwind CSS frontend.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic v2, `regex`, `tiktoken`
* **Frontend:** React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons
* **Deployment:** Render (API Service), Vercel (Static Web App)

---

## 📂 Project Structure

```text
token-intelligence/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── scratch_bpe.py       # Custom BPE trainer & encoder
│   │   │   ├── benchmarks.py        # tiktoken wrappers & token boundaries
│   │   │   ├── pricing_engine.py    # Multi-model cost calculations
│   │   │   └── optimizer.py         # Prompt compression & filler removal
│   │   └── schemas.py               # Pydantic request/response models
│   ├── main.py                      # FastAPI application & endpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Main dashboard & token visualizer
│   │   └── index.css                # Tailwind CSS imports
│   ├── package.json
│   └── vite.config.ts
└── README.md