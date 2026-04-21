#!/bin/bash
# NodeTorch Launcher
# Run this to set up and start NodeTorch.
# First run installs dependencies (~2-5 minutes). After that, starts in seconds.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}🔥 NodeTorch${NC}"
echo ""

# ─────────────────────────────────────────────
# Check Python
# ─────────────────────────────────────────────
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" = "3" ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}✗ Python 3.10+ not found${NC}"
    echo ""
    echo "  Install Python:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    brew install python@3.12"
        echo "    or download from https://python.org/downloads"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "    sudo apt install python3.12 python3.12-venv  (Ubuntu/Debian)"
        echo "    sudo dnf install python3.12  (Fedora)"
        echo "    sudo pacman -S python  (Arch)"
    else
        echo "    Download from https://python.org/downloads"
    fi
    exit 1
fi
echo -e "${GREEN}✓${NC} Python: $($PYTHON --version)"

# ─────────────────────────────────────────────
# Check Node.js
# ─────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo -e "${RED}✗ Node.js not found${NC}"
    echo ""
    echo "  Install Node.js:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    brew install node"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "    sudo apt install nodejs npm  (Ubuntu/Debian)"
        echo "    sudo pacman -S nodejs npm  (Arch)"
    fi
    echo "    or download from https://nodejs.org"
    exit 1
fi
echo -e "${GREEN}✓${NC} Node.js: $(node --version)"

if ! command -v npm &>/dev/null; then
    echo -e "${RED}✗ npm not found (comes with Node.js)${NC}"
    exit 1
fi

# ─────────────────────────────────────────────
# Setup Python venv (first run)
# ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo ""
    echo -e "${YELLOW}First run — setting up Python environment...${NC}"

    $PYTHON -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        echo ""
        echo "  On Ubuntu/Debian, you may need:"
        echo "    sudo apt install python3.12-venv"
        exit 1
    fi

    .venv/bin/pip install --upgrade pip -q

    # Detect GPU and install the right PyTorch
    echo "  Detecting GPU..."
    HAS_CUDA=false
    HAS_MPS=false

    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$CUDA_VER" ]; then
            HAS_CUDA=true
            echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected (driver $CUDA_VER)"
        fi
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        CHIP=$(/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if [[ "$CHIP" == *"Apple"* ]]; then
            HAS_MPS=true
            echo -e "  ${GREEN}✓${NC} Apple Silicon detected (MPS acceleration)"
        fi
    fi

    if [ "$HAS_CUDA" = true ]; then
        echo "  Installing PyTorch with CUDA support..."
        .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
    elif [ "$HAS_MPS" = true ]; then
        echo "  Installing PyTorch with MPS support..."
        .venv/bin/pip install torch torchvision -q
    else
        echo -e "  ${YELLOW}No GPU detected — installing CPU-only PyTorch${NC}"
        echo "  (Training will be slower but everything works)"
        .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to install PyTorch${NC}"
        echo "  Try installing manually:"
        echo "    .venv/bin/pip install torch torchvision"
        exit 1
    fi

    # Install remaining dependencies
    echo "  Installing other dependencies..."
    .venv/bin/pip install -r requirements.txt -q
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        echo "  Check requirements.txt and try:"
        echo "    .venv/bin/pip install -r requirements.txt"
        exit 1
    fi

    echo -e "  ${GREEN}✓${NC} Python dependencies installed"
fi

# ─────────────────────────────────────────────
# Setup frontend (always runs to pick up new deps)
# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
npm install --silent 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ npm install failed${NC}"
    echo "  Try running manually: npm install"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Frontend dependencies installed"

# ─────────────────────────────────────────────
# Check GPU status
# ─────────────────────────────────────────────
GPU_STATUS=$(.venv/bin/python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print(f'CUDA: {name}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon)')
else:
    print('CPU only')
" 2>/dev/null || echo "CPU only")
echo -e "${GREEN}✓${NC} Device: $GPU_STATUS"

# ─────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${BLUE}Frontend${NC}  →  http://localhost:5173"
echo -e "  ${BLUE}Backend${NC}   →  http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Start backend
cd backend
../".venv/bin/python" main.py &
BACKEND_PID=$!
cd "$DIR"

# Wait for backend to be ready
for i in {1..15}; do
    if curl -s http://localhost:8000/system-info >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Start frontend
npm run dev -- --open &
FRONTEND_PID=$!

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

wait
