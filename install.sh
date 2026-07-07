#!/bin/sh
# Sparsify installer — one command from checkout to `sparsify run`.
#
#   ./install.sh                       (from a cloned repo)
#   curl -fsSL <repo>/install.sh | sh  (standalone; set SPARSIFY_REPO first
#                                       if the default below is wrong)
#
# What it does, transparently:
#   1. verifies macOS on Apple Silicon and Python >= 3.10
#   2. installs Sparsify into its own venv at ~/.sparsify/venv
#   3. links the `sparsify` command into ~/.local/bin (or /usr/local/bin)
#   4. leaves models in ~/.sparsify/models (override: SPARSIFY_MODELS_DIR)
set -eu

SPARSIFY_HOME="${SPARSIFY_HOME:-$HOME/.sparsify}"
SPARSIFY_REPO="${SPARSIFY_REPO:-https://github.com/seamlessassist/sparsify}"

say()  { printf '\033[1;36m>>\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31mERROR:\033[0m %s\n' "$1" >&2; exit 1; }

# 1 ── platform checks ------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || fail "Sparsify's MLX backend requires macOS (Apple Silicon). Linux/CUDA is on the roadmap."
[ "$(uname -m)" = "arm64" ]  || fail "Apple Silicon (arm64) required — MLX does not run on Intel Macs."

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$cand"; break
    fi
  fi
done
[ -n "$PY" ] || fail "Python >= 3.10 not found. Install it (e.g. 'brew install python') and re-run."
say "Using $($PY --version 2>&1)"

# 2 ── source tree ----------------------------------------------------------
if [ -f "pyproject.toml" ] && grep -q '^name = "sparsify"' pyproject.toml 2>/dev/null; then
  SRC_DIR="$(pwd)"
  say "Installing from checkout: $SRC_DIR"
else
  command -v git >/dev/null 2>&1 || fail "git not found (needed to fetch Sparsify)."
  SRC_DIR="$SPARSIFY_HOME/src"
  if [ -d "$SRC_DIR/.git" ]; then
    say "Updating existing source in $SRC_DIR"
    git -C "$SRC_DIR" pull --ff-only
  else
    say "Cloning $SPARSIFY_REPO"
    mkdir -p "$SPARSIFY_HOME"
    git clone --depth 1 "$SPARSIFY_REPO" "$SRC_DIR"
  fi
fi

# 3 ── venv -----------------------------------------------------------------
say "Creating venv at $SPARSIFY_HOME/venv"
mkdir -p "$SPARSIFY_HOME"
"$PY" -m venv "$SPARSIFY_HOME/venv"
"$SPARSIFY_HOME/venv/bin/pip" install --quiet --upgrade pip
say "Installing Sparsify (mlx, mlx-lm and friends — a few minutes on first run)"
"$SPARSIFY_HOME/venv/bin/pip" install --quiet "$SRC_DIR[all]" huggingface_hub hf_transfer

# 4 ── launcher -------------------------------------------------------------
BIN_DIR="$HOME/.local/bin"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *)
  if [ -w /usr/local/bin ]; then BIN_DIR=/usr/local/bin; fi ;;
esac
mkdir -p "$BIN_DIR"
ln -sf "$SPARSIFY_HOME/venv/bin/sparsify" "$BIN_DIR/sparsify"
say "Linked sparsify -> $BIN_DIR/sparsify"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) printf '\033[1;33mNOTE:\033[0m add %s to your PATH, e.g.:\n  echo '\''export PATH="%s:$PATH"'\'' >> ~/.zshrc\n' "$BIN_DIR" "$BIN_DIR" ;;
esac

# 5 ── smoke check ----------------------------------------------------------
"$SPARSIFY_HOME/venv/bin/sparsify" --version >/dev/null || fail "installed CLI failed to run"

printf '\n\033[1;32mSparsify installed.\033[0m Try:\n\n'
printf '  sparsify models              # browse the catalog\n'
printf '  sparsify pull olmoe:1b-7b    # 3.9 GB starter MoE\n'
printf '  sparsify run  olmoe:1b-7b    # chat, auto RAM budget\n\n'
printf 'Models live in %s (override with SPARSIFY_MODELS_DIR).\n' "${SPARSIFY_MODELS_DIR:-$SPARSIFY_HOME/models}"
