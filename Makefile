SHELL := /usr/bin/env bash

PYTHON       ?= python
MCP_PID_FILE := .mcp_server.pid

.PHONY: help
help:
	@echo "Make targets:"
	@echo "  make test                   - Rebuild vector DB and run pytest"
	@echo "  make index                  - Clear and rebuild vector database"
	@echo "  make agent SOURCE=...       - Run the multi-agent pipeline on a given article (path or URL)"
	@echo "  make mcp                    - Start MCP server in background"
	@echo "  make stop-mcp               - Stop background MCP server"

.PHONY: test
test:
	@echo "[INFO] Running tests..."
	@PROJECT_ROOT="$$PWD"; \
	if [ -f "$$PROJECT_ROOT/.env" ]; then \
	  echo "[INFO] Loading environment variables from .env"; \
	  set -a; \
	  . "$$PROJECT_ROOT/.env"; \
	  set +a; \
	fi; \
	export PYTHONPATH="$$PROJECT_ROOT:$${PYTHONPATH:-}"; \
	if [ -z "$$GROQ_API_KEY" ]; then \
	  echo "[ERROR] GROQ_API_KEY is not set. Please export it or add it to .env."; \
	  exit 1; \
	fi; \
	echo "[INFO] Clearing and rebuilding vector database before tests..."; \
	rm -rf chroma_db; \
	$(PYTHON) -m scripts.database_ingestion; \
	echo "[INFO] Running pytest..."; \
	pytest -q; \
	echo "[OK] All tests finished."

.PHONY: index
index:
	@echo "[INFO] Clearing existing vector database (chroma_db)..."
	@rm -rf chroma_db/*
	@echo "[INFO] Rebuilding vector database..."
	@PROJECT_ROOT="$$PWD"; \
	export PYTHONPATH="$$PROJECT_ROOT:$${PYTHONPATH:-}"; \
	$(PYTHON) -m scripts.database_ingestion; \
	echo "[OK] Vector database rebuilt."

.PHONY: agent
agent:
	@if [ -z "$(SOURCE)" ]; then \
		echo "[ERROR] SOURCE is not set."; \
		echo "Usage: make agent SOURCE='samples/input_article_1.pdf'"; \
		echo "   or: make agent SOURCE='https://example.com/file.pdf'"; \
		exit 1; \
	fi; \
	echo "[INFO] Running pipeline for SOURCE='$(SOURCE)'..."; \
	PROJECT_ROOT="$$PWD"; \
	if [ -f "$$PROJECT_ROOT/.env" ]; then \
	  echo "[INFO] Loading environment variables from .env"; \
	  set -a; \
	  . "$$PROJECT_ROOT/.env"; \
	  set +a; \
	fi; \
	export PYTHONPATH="$$PROJECT_ROOT:$${PYTHONPATH:-}"; \
	if [ -z "$$GROQ_API_KEY" ]; then \
	  echo "[ERROR] GROQ_API_KEY is not set. Please export it or add it to .env."; \
	  exit 1; \
	fi; \
	echo "[INFO] Running agents on SOURCE='$(SOURCE)'..."; \
	$(PYTHON) -m scripts.run_agents "$(SOURCE)"; \
	echo "[OK] Sample run finished."

.PHONY: mcp
mcp:
	@echo "[INFO] Starting MCP server in background..."
	@PROJECT_ROOT="$$PWD"; \
	export PYTHONPATH="$$PROJECT_ROOT:$${PYTHONPATH:-}"; \
	nohup $(PYTHON) -m src.mcp_server.server >/tmp/mcp_server.log 2>&1 & echo $$! > "$(MCP_PID_FILE)"; \
	echo "[OK] MCP server started with PID $$(cat $(MCP_PID_FILE)). Logs: /tmp/mcp_server.log"

.PHONY: stop-mcp
stop-mcp:
	@if [ ! -f "$(MCP_PID_FILE)" ]; then \
		echo "[WARN] No MCP PID file found at $(MCP_PID_FILE). Nothing to stop."; \
		exit 0; \
	fi; \
	PID=$$(cat $(MCP_PID_FILE)); \
	if kill -0 $$PID 2>/dev/null; then \
		echo "[INFO] Stopping MCP server (PID $$PID)..."; \
		kill $$PID || true; \
	else \
		echo "[WARN] MCP server process $$PID not running."; \
	fi; \
	rm -f "$(MCP_PID_FILE)"; \
	echo "[OK] MCP server stopped (or was already not running)."
