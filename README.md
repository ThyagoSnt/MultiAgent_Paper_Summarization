# Multi-Agent System for Paper Summary Tasks (Multi-Agent + Vector Store + OCR + MCP)

This repository implements the technical challenge of building a multi-agent system that:

1. Receives a paper (local PDF, URL, or text).
2. Classifies the paper into one of three scientific areas.
3. Extracts a JSON with fields **exactly** matching the prompt (including the *typo* in `artcle`).
4. Produces a critical review in Portuguese, highlighting strengths, limitations, and threats to validity.

The system is composed of:

- A **vector store** in ChromaDB with 9 papers (3 areas × 3 PDFs per area).
- An **MCP server** exposing access to that vector store (`search_articles`, `get_article_content`).
- A **multi-agent pipeline** in LangGraph (classifier → extractor → reviewer).
- Optional **OCR support** via Docling when the PDF has no embedded text.
- **Automated tests** for the vector store and the integrated pipeline.
- **Makefile**, **Dockerfile**, and **docker-compose** to simplify setup and execution.

---

## 1. High-level architecture

Summary flow:

1. **Vector database ingestion**
   - PDFs organized under `pdf_database/<area>/` (e.g., `economy`, `med`, `tech`).
   - Each PDF is split into *chunks* (`chunk_size=1000`, `overlap=200`).
   - Embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`.
   - Everything is indexed in a Chroma collection named `articles` under `chroma_db/`.

2. **MCP Server (`src/mcp_server/server.py`)**
   - Reads configuration from `configuration/base.yaml`.
   - Opens the existing Chroma index.
   - Exposes two tools via MCP:
     - `search_articles(query: str, top_k: int)` → list of `{id, title, area, score}`.
     - `get_article_content(article_id: str)` → `{id, title, area, content}`.

3. **Multi-agent system (`src/multi_agent_system`)**
   - Implemented with **LangGraph**.
   - Main nodes:
     - **classifier_node**  
       Classifies the paper into an area.  
       Uses:
       - The paper text (truncated to 4000 characters).
       - Context retrieved via MCP (`search_articles` with an initial snippet).
     - **extractor_node**  
       Reads the paper (with safety truncation) and fills a JSON with the schema:

       ```json
       {
         "what problem does the artcle propose to solve?": "",
         "step by step on how to solve it": ["", "", ""],
         "conclusion": ""
       }
       ```

       The keys are **identical** to the prompt, including `artcle`.
     - **reviewer_node**  
       Uses the extracted JSON + part of the paper text to produce a critical review in Portuguese (Markdown).

   - The graph is:  
     `start → classifier → extractor → reviewer → END`.

4. **User input (`scripts/run_agents.py` + `src/pipeline/pipeline_runner.py`)**
   - Accepts:
     - Local path to `.pdf`, `.txt`, or `.md`.
     - A URL pointing to a PDF.
   - Normalizes the input and saves it to `samples/input_article_N.ext`.
   - Runs the multi-agent pipeline (`run_pipeline`).
   - Generates:
     - `samples/review_N.md` with the review.
     - `samples/output_N.json` with `{area, extraction, review_markdown}`.

---

## 2. Directory structure

Overview:

```text
.
├── chroma_db/               # Where the persistent ChromaDB vector index is stored
├── configuration/
│   └── base.yaml            # Central configuration file (paths, vector DB, MCP, LLM, etc.)
├── pdf_database/            # Local PDF base used as reference for the vector store
│   ├── economy/             # 3 economy papers
│   ├── med/                 # 3 medical papers
│   └── tech/                # 3 technology papers
├── samples/                 # Outputs generated when running the pipeline
│   ├── input_article_N.pdf  # Normalized inputs (PDF/URL/MD/TXT)
│   ├── output_N.json        # Structured output (area + extracted JSON + review)
│   └── review_N.md          # Critical review written by the agent
├── src/
│   ├── mcp_server/
│   │   └── server.py        # MCP server exposing tools backed by the vector store
│   ├── multi_agent_system/
│   │   ├── classifier_agent.py   # Agent that classifies the paper's area
│   │   ├── extractor_agent.py    # Agent that generates the JSON required by the challenge
│   │   ├── reviewer_agent.py     # Agent that writes the critical review
│   │   ├── graph.py              # LangGraph graph connecting the three agents
│   │   └── mcp_vector_client.py  # HTTP client that talks to the MCP server
│   ├── pdf_parser/
│   │   └── pdf_parser.py         # PDF reader (PyPDF + Docling OCR fallback when needed)
│   ├── pipeline/
│   │   └── pipeline_runner.py    # Orchestrates the full execution flow
│   └── vector_database/
│       ├── vector_database.py    # Vector store implementation (Chroma + embeddings)
│       └── ingestion_runner.py   # Handles ingestion of the 9 PDFs and index creation
├── scripts/
│   ├── database_ingestion.py     # Script to rebuild the vector database
│   └── run_agents.py             # CLI script to run the pipeline on a paper
├── tests/                        # Automated test suite
│   ├── test_vector_database.py
│   └── test_graph_pipeline.py
├── mcp.json                      # MCP manifest (for external MCP clients)
├── Makefile                      # Main commands (setup, index, tests, agents, mcp)
├── Dockerfile                    # Project Docker image
├── docker-compose.yml            # Docker Compose orchestration
├── requirements.txt              # Python dependencies
└── pytest.ini                    # Pytest configuration
```

---

## 3. Requirements

### 3.1. Main dependencies

Execution is designed to run **inside a Docker container**. On the host, you only need:

- Docker
- Docker Compose
- An environment variable or `.env` file with:

  ```bash
  GROQ_API_KEY=<your_groq_key>
  ```

Optionally, if you run on a machine with GPU and CUDA configured, PyTorch and `sentence-transformers` will use it automatically inside the container.

### 3.2. Environment variables

At the project root, create a `.env` with:

```bash
echo "GROQ_API_KEY=YOUR_TOKEN_HERE" > .env
```

`docker-compose.yml` is already configured to load this `.env` into the container.

---

## 4. Running with Docker and docker-compose

All dependency installation, vector index creation, and pipeline execution happen **inside** the container.

### 4.1. Build the image

At the project root:

```bash
docker compose build
```

### 4.2. Enter the container

To open a shell inside the container:

```bash
docker compose run --rm summary-app bash
```

You will be in `/app` inside the container, with the code mounted via a volume.

---

## 5. Building the vector store (ingesting the 9 papers)

![Index build diagram](./documentation_images/index.png)

Still **inside the container**, run:

```bash
make index
```

The `index` target:

1. Cleans the `chroma_db/` folder (if it exists).
2. Rebuilds the full vector index by calling `scripts.database_ingestion`.

Internally it:

1. Reads `configuration/base.yaml`.
2. Resolves `pdf_database/` and `chroma_db/`.
3. Creates a `VectorDatabase` instance with:
   - `embedding_model="sentence-transformers/all-MiniLM-L6-v2"`
   - `chunk_size=1000`
   - `chunk_overlap=200`
4. Iterates each subfolder under `pdf_database/`:
   - Each PDF is read, text extracted, chunked, and indexed.
5. Persists embeddings and metadata into Chroma (stored under `/app/chroma_db`, which is mounted to the host).

---

## 6. OCR and text extraction from PDFs

![Parser diagram](./documentation_images/parser.png)

The `PdfTextExtractor` class (`src/pdf_parser/pdf_parser.py`) implements:

1. **Extraction with PyPDF**
   - Uses `pypdf.PdfReader` and `page.extract_text()` page by page.
   - If it can extract text, it returns that content.

2. **OCR fallback via Docling (optional)**
   - If **no text** is found with PyPDF and `enable_ocr=True`, it tries OCR.
   - Uses `docling.document_converter.DocumentConverter().convert(...)`.
   - Calls `export_to_text()`.
   - If there is still no text, it raises `ValueError`.

In the pipeline, OCR is enabled by default for `.pdf` files.

---

## 7. Multi-agent pipeline (LangGraph)

![Agent system diagram](./documentation_images/agent.png)

The graph lives in `src/multi_agent_system/graph.py`. Components:

- **Classifier Agent (`classifier_agent.py`)**
  - Discovers the areas by listing subfolders under `pdf_database/` (e.g., `economy`, `med`, `tech`).
  - Truncates the paper text to `max_article_chars` (4000 characters).
  - Uses the MCP client (`MCPVectorStoreClient`) to call `search_articles` with an initial snippet (`mcp_query_chars`, 800 characters).
  - Builds a textual context from the vector store hits.
  - Calls the Groq LLM with:
    - A system prompt configured in `MultiAgentConfig`.
    - A human message containing truncated paper text + similar-paper list.
  - Normalizes the output into a known area (exact match, substring, synonyms like “econ”).

- **Extractor Agent (`extractor_agent.py`)**
  - Receives `article_text` (and optionally `area`).
  - Truncates the text to `max_article_chars` (6000).
  - Instructs the LLM to return **only** JSON.
  - Uses `_extract_json_from_response` to support:
    - Raw JSON.
    - ```json ... ``` fenced blocks.
  - Uses `_normalize_extraction` to ensure:

    ```json
    {
      "what problem does the artcle propose to solve?": "string",
      "step by step on how to solve it": ["string", "..."],
      "conclusion": "string"
    }
    ```

- **Reviewer Agent (`reviewer_agent.py`)**
  - Receives `area`, `extraction`, and `article_text`.
  - Serializes `extraction` to a JSON string.
  - Truncates the paper to ~4000 characters for optional context.
  - Generates a critical review in Portuguese, focusing on:
    - Novelty / contribution.
    - Method / experimental design.
    - Validity of results.
    - Threats to validity and reproducibility.

- **Graph (`graph.py`)**
  - State: `{"article_text", "area", "extraction", "review"}`.
  - Order:
    - `start → classifier → extractor → reviewer → END`.
  - Public function:
    - `run_pipeline(article_text: str) -> Dict[str, Any]`.

---

## 8. MCP server

![MCP system diagram](./documentation_images/mcp.png)

In `src/mcp_server/server.py`:

- Reads `configuration/base.yaml`:
  - Configures `pdf_root`, `chroma_path`.
  - Configures `embedding_model`, `collection_name`, `chunk_size`, `chunk_overlap`.
  - Configures `mcp.name`, `mcp.transport`.
- Instantiates `VectorDatabase` with these settings.
- Creates a `FastMCP` server with `json_response=True`.
- Exposed tools:

```python
@mcp.tool()
def search_articles(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    ...
```

```python
@mcp.tool()
def get_article_content(article_id: str) -> Dict[str, Any]:
    ...
```

### 8.1. MCP Manifest (`mcp.json`)

`mcp.json` defines:

- Server name, label, version, and description.
- How to start it via STDIO:

```jsonc
{
  "servers": [
    {
      "name": "stdio",
      "transport": "stdio",
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  ],
  "tools": [
    { "name": "search_articles", ... },
    { "name": "get_article_content", ... }
  ]
}
```

This is used by external MCP clients (e.g., ChatGPT or IDEs) to discover how to talk to the server inside the container.

### 8.2. Internal MCP client (`MCPVectorStoreClient`)

`src/multi_agent_system/mcp_vector_client.py`:

- Uses `mcp.ClientSession` + `stdio_client` to launch the MCP server as a subprocess:
  - Default command: `python -m src.mcp_server.server 2>/dev/null`.
- Public methods:
  - `search_articles(query: str, top_k: int)`.
  - `get_article_content(article_id: str)`.

In other words, internal agents do not access `VectorDatabase` directly; everything goes through the MCP tools.

---

## 9. Pipeline execution CLI (use the Makefile explained next)

`src/pipeline/pipeline_runner.py`:

- `ArticleSampleManager`:
  - Reads `samples/output_N.json` to discover the next index.
  - Copies local files or downloads URLs into `samples/input_article_N.ext`.
  - Computes paths for `review_N.md` and `output_N.json`.

- `ArticlePipelineRunner`:
  - Resolves source → normalizes input → reads text.
  - Calls `run_pipeline(article_text)`.
  - Saves `review_N.md` and `output_N.json`.
  - Returns execution metadata.

---

## 10. Automated tests

### 10.1. Running tests (inside the container)

With `GROQ_API_KEY` available in the environment (via `.env`):

```bash
make test
```

The `test` target:

1. Rebuilds the vector database (`scripts.database_ingestion`).
2. Runs `pytest`.

### 10.2. Tests

- `tests/test_vector_database.py`:
  - Tests chunking, `search_articles`, and `get_article_content`.
- `tests/test_graph_pipeline.py`:
  - End-to-end integration of the graph (`run_pipeline`) on a PDF.
  - Skips if `GROQ_API_KEY` is not set.

---

## 11. Makefile

Main targets available **inside the container**:

```text
make help
```

- `make index`  
  Cleans the `chroma_db/` folder and rebuilds the full vector index from `pdf_database/`.

- `make test`  
  Rebuilds the index and runs `pytest`.

- `make agent SOURCE=...`  
  Runs the multi-agent pipeline on a paper (local file or URL).  
  Examples:

  ```bash
  make agent SOURCE="samples/input_article_1.pdf"
  make agent SOURCE="https://example.com/paper.pdf"
  ```

- `make mcp`  
  Starts the MCP server in the background inside the container, writing the PID to `.mcp_server.pid`. This is only useful if you want to connect an external MCP client (like ChatGPT) to the server running inside the container.

- `make stop-mcp`  
  Stops the MCP server started via `make mcp`.

---

## 12. Docker and docker-compose

Summary of the Docker flow:

1. **Build the image** (on the host):

   ```bash
   docker compose build
   ```

2. **Enter the container** (on the host):

   ```bash
   docker compose run --rm summary-app bash
   ```

3. **Inside the container** (in `/app`):

   ```bash

   # 1) Build the vector index in chroma_db/
   make index

   # 2) Run the pipeline on an example paper
   make agent SOURCE="samples/input_article_1.pdf"

   # 3) Run tests
   make test

   # 4) Start MCP server in the background for external clients
   make mcp

   # ... use it from an external MCP client ...
   make stop-mcp
   ```

The `/app/chroma_db`, `/app/samples`, and `/app/pdf_database` folders are mounted as host volumes, so artifacts generated inside the container appear in the local project tree.

---

## 13. Quick summary

1. Create `.env` with the key (on the host):

   ```bash
   echo "GROQ_API_KEY=YOUR_TOKEN_HERE" > .env
   ```

2. Build the Docker image (on the host):

   ```bash
   docker compose build
   ```

3. Enter the container:

   ```bash
   docker compose run --rm summary-app bash
   ```

4. Inside the container, build the index:

   ```bash
   make index
   ```

5. Run the server and the pipeline on a sample:

   ```bash
   make mcp
   make agent SOURCE="samples/input_article_1.pdf"
   ```

6. Check outputs (from host or container):

   - `samples/output_N.json`
   - `samples/review_N.md`

---

# 14. Detailed configuration (`base.yaml`) — parameter explanations

Below is a clear and objective explanation of **what each configuration field does** and **why it was chosen** within this system.

## **`mcp` block: MCP server parameters**

```yaml
mcp:
  name: "ArticleVectorStore"
  transport: "http"
  host: "0.0.0.0"
  port: 8000
  base_url: "http://127.0.0.1:8000"
```

### ✦ `name: "ArticleVectorStore"`
Logical name of the MCP server.  
Used only for identification by external MCP clients.  
We chose this name because the server exposes exactly one vector store of papers.

### ✦ `transport: "http"`
Defines that MCP runs as an **HTTP server**, not STDIO.  
This allows the multi-agent system to talk to MCP using normal HTTP requests — simpler, more predictable, and easier to debug.

### ✦ `host: "0.0.0.0"`
Makes the server listen on all network interfaces inside the container.  
This is required so the multi-agent system can reach MCP even when running in separate processes.

### ✦ `port: 8000`
Default port for the project's FastAPI/uvicorn server.

### ✦ `base_url: "http://127.0.0.1:8000"`
URL used by the MCP client (`MCPVectorStoreClient`) to send requests.  
Even though the server binds to `0.0.0.0`, the client connects to `127.0.0.1` from within the container.

## **`paths` block: main paths**

```yaml
paths:
  pdf_root: pdf_database
  chroma_path: chroma_db
```

### ✦ `pdf_root`
Directory containing the 9 PDFs used to build the vector store.  
We chose a `<area>/<pdf>` structure because the classifier discovers areas automatically by listing the subfolders.

### ✦ `chroma_path`
Directory where ChromaDB saves its persistent vector database.  
Kept simple and at the project root to simplify versioning and cleanup via `make index`.

## **`vector_db` block: vector store parameters**

```yaml
vector_db:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  collection_name: articles
  chunk_size: 1000
  chunk_overlap: 200
```

### ✦ `embedding_model: all-MiniLM-L6-v2`
A fast, lightweight model that is widely recommended for **semantic similarity**.  
Motivations:
- Much lighter than large models
- Strong embeddings for shorter technical text
- Low latency, ideal for multi-agent pipelines

### ✦ `collection_name: articles`
ChromaDB collection name.  
We kept `articles` because the stored content is exclusively scientific papers.

### ✦ `chunk_size: 1000`  
Each PDF is split into chunks of up to 1000 characters.  
This size balances:
- enough context per chunk  
- good granularity for vector search  
- faster embedding generation  

### ✦ `chunk_overlap: 200`  
20% overlap between chunks.  
Chosen to prevent context loss across chunk boundaries — especially important for longer PDFs.

## **`multi_agent` block: LLM agent configuration**

```yaml
multi_agent:
  llm:
    provider: groq
    model: openai/gpt-oss-120b
    temperature: 0.0
```

### ✦ `provider: groq`
We use **Groq** for extremely low latency, reducing overall pipeline response time.

### ✦ `model: openai/gpt-oss-120b`
An OSS model accelerated by Groq, with a strong balance of:
- cost
- speed
- reasoning capability

Empirically, it produced the most consistent results for the classifier, extractor, and reviewer.

### ✦ `temperature: 0.0`
Fully deterministic behavior.  
Essential for:
- automated tests being stable  
- JSON extraction without structural variation  
- classification not oscillating across runs

## **Agent prompts**

### **Classifier prompt**
Focused on selecting **exactly one** of the three areas.  
It has access to the output of `search_articles()`, encouraging use of real vector-store context.

### **Extractor prompt**
Forces the LLM to return **only** the JSON with:
- keys identical to the prompt
- strict format
- an assertive step-by-step solution

The `artcle` typo is preserved to follow the official prompt.

### **Reviewer prompt**
Generates a complete critical review in **Brazilian Portuguese**, with eight mandatory sections.  
It also guides the model to recognize when the paper is “out of area,” matching the classifier behavior.

## **Main dependencies and rationale**

### **groq**
Low-latency LLM provider → reduces pipeline response time.

### **chromadb**
Simple and efficient vector store with a direct Python API.

### **sentence-transformers**
High-quality embeddings for semantic search.

### **pypdf**
First-stage text extraction — fast and without native dependencies.

### **docling**
Modern OCR used as a fallback for PDFs without embedded text.

### **easyocr**
Complementary dependency used internally by Docling in some scenarios.

### **langgraph**
Builds a deterministic and reproducible agent pipeline.

### **langchain-groq**
Groq driver to simplify LLM calls.

### **fastapi + uvicorn**
Lightweight infra for the MCP server over HTTP.

### **pydantic**
Strict validation for all MCP requests/responses.

### **pytest**
Automated tests for the vector store and the multi-agent pipeline.