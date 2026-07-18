# Sparsify GitHub Issues Catalog (25 Actionable Task Templates)

This document contains 25 pre-formatted GitHub issues ready to be copy-pasted into your repository. They are organized by component and categorized by difficulty level (including `good first issue` tags) to attract and guide contributors.

---

## 🎨 Web UI & Front-end

### 1. [Web UI] Add custom dark/light theme toggle
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `webui`, `enhancement`
*   **Files involved**: [site/](file:///Volumes/projects/sparsify/site)
*   **Description**: Currently, the Web UI defaults to dark mode. We should add a persistent theme toggle button in the sidebar that allows users to switch between light and dark mode, storing the preference in `localStorage`.
*   **Steps to implement**:
    1. Add a theme toggle button or icon (sun/moon) to the sidebar HTML/JS.
    2. Write CSS classes for light mode theme variables (background colors, text colors, borders).
    3. Implement a JS helper to toggle the `.light-mode` class on the `<body>` element and save it to `localStorage`.

### 2. [Web UI] Draw real-time cache hit rate charts with SVG/CSS
*   **Difficulty**: Medium
*   **Recommended Labels**: `webui`, `enhancement`
*   **Files involved**: [site/](file:///Volumes/projects/sparsify/site)
*   **Description**: Telemetry reports hit rates numerically under responses. It would be highly visually appealing to draw a live, smooth line graph or radial progress indicator using SVG to display caching performance.
*   **Steps to implement**:
    1. Create a container in the chat UI metadata panel for the cache chart.
    2. Use standard canvas or SVG to draw/update data points dynamically as telemetry JSON payload arrives via SSE.
    3. Add smooth CSS transitions to updates.

### 3. [Web UI] Make chat interface mobile-responsive
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `webui`, `bug`
*   **Files involved**: [site/](file:///Volumes/projects/sparsify/site)
*   **Description**: When viewing the Web UI on mobile devices or narrow browser windows, the sidebar overlaps with the chat panel or gets truncated.
*   **Steps to implement**:
    1. Add CSS media queries for viewport widths under `768px`.
    2. Hide the sidebar on narrow screens and implement a hamburger menu toggle button.
    3. Adjust padding and text sizes of speech bubbles for phone dimensions.

### 4. [Web UI] Visual model manager inside sidebar
*   **Difficulty**: Medium
*   **Recommended Labels**: `webui`, `enhancement`
*   **Files involved**: [site/](file:///Volumes/projects/sparsify/site)
*   **Description**: Users currently pull and remove models via CLI or the web UI select box. We should add a dedicated "Models" tab/view in the sidebar that lists all models in our catalog with status tags (`Ready` / `Not Installed`), and click-to-download/remove buttons.
*   **Steps to implement**:
    1. Create a visual model list view in HTML/CSS.
    2. Add REST endpoints for `/api/models/pull` and `/api/models/remove` in `server.py`.
    3. Connect frontend buttons to trigger fetches and display download progress bars.

### 5. [Web UI] Export chats to Markdown or PDF
*   **Difficulty**: Easy
*   **Recommended Labels**: `webui`, `enhancement`
*   **Files involved**: [site/](file:///Volumes/projects/sparsify/site)
*   **Description**: Add an option in the Web UI to download active chat transcripts as formatted Markdown (`.md`) or text files.
*   **Steps to implement**:
    1. Create an "Export Chat" dropdown button near the chat header.
    2. Parse the active chat history DOM/state into markdown text.
    3. Generate a dynamic download file link using client-side JavaScript.

---

## 📟 Terminal UI (TUI)

### 6. [TUI] Prevent terminal crash on window resize
*   **Difficulty**: Medium
*   **Recommended Labels**: `tui`, `bug`
*   **Files involved**: [src/sparsify/cli.py](file:///Volumes/projects/sparsify/src/sparsify/cli.py)
*   **Description**: Resizing the terminal window during an active `sparsify run` chat session can sometimes cause layout exceptions in `prompt_toolkit`.
*   **Steps to implement**:
    1. Hook into window resize events inside the `prompt_toolkit` generation loop.
    2. Recalculate terminal rows/columns and trigger a clean UI redraw.
    3. Handle exceptions caused by shrinking the window past UI boundary limits.

### 7. [TUI] Add mouse scroll wheel support in terminal
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `tui`, `enhancement`
*   **Files involved**: [src/sparsify/cli.py](file:///Volumes/projects/sparsify/src/sparsify/cli.py)
*   **Description**: Mouse scrolling is currently disabled in the full-screen terminal TUI. Enabling mouse scroll support would allow users to look back at past chat history.
*   **Steps to implement**:
    1. Enable mouse support in `prompt_toolkit` config parameters.
    2. Bind mouse scroll events to scroll the text window viewport.

### 8. [TUI] TUI keyboard shortcuts help modal
*   **Difficulty**: Easy
*   **Recommended Labels**: `tui`, `enhancement`
*   **Files involved**: [src/sparsify/cli.py](file:///Volumes/projects/sparsify/src/sparsify/cli.py)
*   **Description**: Users don't know the keys to exit, clear, or manage terminal chat. Pressing `?` or `Ctrl+H` should trigger a clean help pop-up listing all TUI shortcuts.
*   **Steps to implement**:
    1. Listen for help shortcuts inside TUI input handlers.
    2. Render a simple modal box centered in the console listing key combinations.

### 9. [TUI] Code block syntax highlighting in bubble replies
*   **Difficulty**: Medium
*   **Recommended Labels**: `tui`, `enhancement`
*   **Files involved**: [src/sparsify/cli.py](file:///Volumes/projects/sparsify/src/sparsify/cli.py)
*   **Description**: While Markdown is parsed, code blocks (e.g. ```python) lack syntax coloring inside the terminal speech bubbles.
*   **Steps to implement**:
    1. Integrate `pygments` or `rich.syntax` styling helper.
    2. Detect code fences inside token reply streams and style code blocks on the fly before drawing.

### 10. [TUI] Long words overflow wrapping bug in narrow terminals
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `tui`, `bug`
*   **Files involved**: [src/sparsify/cli.py](file:///Volumes/projects/sparsify/src/sparsify/cli.py)
*   **Description**: If a generated response contains long links or uninterrupted lines (like file paths), the speech bubble breaks bounds and shifts terminal lines.
*   **Steps to implement**:
    1. Add custom word wrapping or truncation boundaries based on window size.
    2. Wrap long paths cleanly inside bubbles using text-wrap rules.

---

## 🗄️ Database & Storage

### 11. [Database] Export profiling history to CSV/JSON format
*   **Difficulty**: Easy
*   **Recommended Labels**: `database`, `enhancement`
*   **Files involved**: [src/sparsify/storage/database.py](file:///Volumes/projects/sparsify/src/sparsify/storage/database.py)
*   **Description**: Add a database query command to dump historical runs, telemetry, and memory profile records into standard CSV or JSON files.
*   **Steps to implement**:
    1. Implement a new CLI handler: `sparsify history --export <csv/json>`.
    2. Fetch all runs from the database and serialize fields cleanly.

### 12. [Database] Automated database indexing and size cleanup
*   **Difficulty**: Medium
*   **Recommended Labels**: `database`, `maintenance`
*   **Files involved**: [src/sparsify/storage/database.py](file:///Volumes/projects/sparsify/src/sparsify/storage/database.py)
*   **Description**: Telemetry history accumulates quickly, bloating the database file size. We need automated purging or size limit warnings.
*   **Steps to implement**:
    1. Add a setting for telemetry retention limits (e.g. 30 days).
    2. Write a cleanup query that runs once daily or at CLI startup.
    3. Run `VACUUM` commands periodically to keep database files compact.

### 13. [Database] Save chat session histories locally
*   **Difficulty**: Hard
*   **Recommended Labels**: `database`, `enhancement`
*   **Files involved**: [src/sparsify/storage/database.py](file:///Volumes/projects/sparsify/src/sparsify/storage/database.py)
*   **Description**: Currently, closing the server or terminal wipes the active chat context. We should save conversation history in SQLite to load them back on restart.
*   **Steps to implement**:
    1. Create a `conversations` database schema: `id`, `session_title`, `model`, `created_at`.
    2. Save message pairs dynamically as they are processed.
    3. Render past sessions in the Web UI left sidebar.

---

## 🧠 Core Paging & Performance (PyTorch / MLX)

### 14. [Core] Multi-threaded safetensors expert loader
*   **Difficulty**: Hard
*   **Recommended Labels**: `core`, `performance`
*   **Files involved**: [src/sparsify/paging_torch/store.py](file:///Volumes/projects/sparsify/src/sparsify/paging_torch/store.py)
*   **Description**: Loading experts sequentially introduces block latencies during model surgery. We should load experts using multiple thread workers.
*   **Steps to implement**:
    1. Implement concurrent ranges reading via Python `concurrent.futures`.
    2. Overlap the retrieval of multiple routed experts in a single routing step.
    3. Measure speed differences across standard SSD hardware.

### 15. [Core] Predictive prefetching of experts
*   **Difficulty**: Hard
*   **Recommended Labels**: `core`, `performance`
*   **Files involved**: [src/sparsify/paging_torch/surgery.py](file:///Volumes/projects/sparsify/src/sparsify/paging_torch/surgery.py)
*   **Description**: The model routing decision is calculated one block layer ahead. We can use this prediction gap to prefetch the experts for layer `N+1` while computing layer `N`.
*   **Steps to implement**:
    1. Hook into the forward pass calculation logic.
    2. Spawn a background thread worker to fetch the next block's experts asynchronously.
    3. Measure reduction in forward pass delays.

### 16. [Core] Reuse memory-mapped file handles
*   **Difficulty**: Medium
*   **Recommended Labels**: `core`, `bug`
*   **Files involved**: [src/sparsify/paging_torch/store.py](file:///Volumes/projects/sparsify/src/sparsify/paging_torch/store.py)
*   **Description**: Opening file handles for every expert read block creates high system descriptor activity. We should pool and reuse file descriptors.
*   **Steps to implement**:
    1. Keep a cache of open file handles for target safetensors shards.
    2. Close handles cleanly during server shut down or model unload.

### 17. [Core] Support GGUF layer offloading alongside PyTorch paging
*   **Difficulty**: Hard
*   **Recommended Labels**: `core`, `performance`
*   **Files involved**: [src/sparsify/backends/](file:///Volumes/projects/sparsify/src/sparsify/backends)
*   **Description**: Explore hybrid paging of GGUF formats directly from memory mapping, allowing CPU layers to remain un-allocated.
*   **Steps to implement**:
    1. Analyze memory mappings for GGUF layouts.
    2. Add custom GGUF tensor paging hook handlers.

### 18. [Core] CPU Matrix Kernel optimizations (Intel/AMD)
*   **Difficulty**: Medium
*   **Recommended Labels**: `core`, `performance`
*   **Files involved**: [src/sparsify/backends/pytorch_backend.py](file:///Volumes/projects/sparsify/src/sparsify/backends/pytorch_backend.py)
*   **Description**: When PyTorch runs on standard CPU backends, inference can be slow. We should investigate using `mkl` or Intel-optimized CPU runtimes.
*   **Steps to implement**:
    1. Detect Intel CPU extensions.
    2. Set thread configurations: `torch.set_num_threads()` based on physical CPU cores.

---

## 🛠️ Installer & Deployment

### 19. [Installer] Detect and configure CUDA automatically on Linux/Windows
*   **Difficulty**: Medium
*   **Recommended Labels**: `installer`, `enhancement`
*   **Files involved**: [install.sh](file:///Volumes/projects/sparsify/install.sh)
*   **Description**: Non-macOS systems need PyTorch with CUDA support. The script should verify CUDA version and install the matching pip wheel version.
*   **Steps to implement**:
    1. Check for `nvcc` or `nvidia-smi` version outputs.
    2. Run pip install targeting the specific PyTorch CUDA wheel URL (e.g. `cu121`/`cu124`).

### 20. [Installer] Docker container setup for server mode
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `installer`, `enhancement`
*   **Files involved**: [Dockerfile](file:///Volumes/projects/sparsify/Dockerfile)
*   **Description**: Provide a Dockerfile and docker-compose script to deploy the Sparsify server inside a containerized host.
*   **Steps to implement**:
    1. Write a clean Multi-stage Dockerfile setup.
    2. Mount the local model folder to prevent redundant downloads.

---

## 🔌 API & Integration

### 21. [API] Add OpenAI-compatible `/v1/embeddings` endpoint
*   **Difficulty**: Medium
*   **Recommended Labels**: `api`, `enhancement`
*   **Files involved**: [src/sparsify/runtime/server.py](file:///Volumes/projects/sparsify/src/sparsify/runtime/server.py)
*   **Description**: Allow external vector search software (like Chroma or Qdrant) to connect directly to Sparsify.
*   **Steps to implement**:
    1. Add `/v1/embeddings` POST route.
    2. Feed tokens to the model backbone, extract the hidden state, and return the vector JSON.

### 22. [API] Add basic auth rate-limiting middleware
*   **Difficulty**: Medium
*   **Recommended Labels**: `api`, `security`
*   **Files involved**: [src/sparsify/runtime/server.py](file:///Volumes/projects/sparsify/src/sparsify/runtime/server.py)
*   **Description**: If the API is exposed on local networks, we need authentication.
*   **Steps to implement**:
    1. Add an optional API Key setting `SPARSIFY_API_KEY`.
    2. Write request checking middleware that blocks unauthorized calls.

---

## 📄 Documentation

### 23. [Docs] Add a detailed guide on setting up Sparsify on WSL2
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `documentation`
*   **Files involved**: [docs/](file:///Volumes/projects/sparsify/docs)
*   **Description**: Guide Windows developers on setting up GPU-passthrough CUDA drivers inside WSL2.
*   **Steps to implement**:
    1. Create `docs/WSL2_SETUP.md`.
    2. Outline step-by-step commands for WSL setup.

### 24. [Docs] Translate project documentation
*   **Difficulty**: Easy
*   **Recommended Labels**: `documentation`
*   **Files involved**: [docs/](file:///Volumes/projects/sparsify/docs)
*   **Description**: Translate README and instructions into popular languages (Spanish, Chinese, Japanese) to grow global adoption.
*   **Steps to implement**:
    1. Create language-specific docs files.
    2. Link translations in the main README.

### 25. [Docs] Update contribution guide and templates
*   **Difficulty**: Easy (`good first issue`)
*   **Recommended Labels**: `documentation`
*   **Files involved**: [CONTRIBUTING.md](file:///Volumes/projects/sparsify/CONTRIBUTING.md)
*   **Description**: Prepare coding style checks, tests setup instruction, and pull request checklist template.
*   **Steps to implement**:
    1. Update contribution documentation with development environment check instructions.
    2. Add standard templates under `.github/PULL_REQUEST_TEMPLATE.md`.
