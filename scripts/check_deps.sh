#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}[INFO]${NC} Checking SparseRNJ dependencies"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================"
echo "SparseRNJ Dependency Check"
echo "================================================"

# Track missing dependencies
missing_deps=()
missing_optional=()

# Check Python dependencies (conda environment should be activated)
echo -e "${BLUE}[INFO]${NC} Checking Python environment..."
if ! python -c "import numpy, dendropy, networkx, pandas, matplotlib, scipy, sklearn, seaborn, tqdm" 2>/dev/null; then
    missing_deps+=("Python packages")
    echo -e "${RED}[ERROR]${NC} Missing Python packages. Please activate conda environment:"
    echo -e "${RED}[ERROR]${NC}   conda env create -f environment.yml"
    echo -e "${RED}[ERROR]${NC}   conda activate srnj"
else
    echo -e "${GREEN}[SUCCESS]${NC} Python packages available"
fi

# Check required external binaries for tree distance computation
echo ""
echo -e "${BLUE}[INFO]${NC} Checking external binaries..."

# Check tqDist binaries (quartet_dist and triplet_dist)
if command -v quartet_dist &> /dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} quartet_dist found: $(which quartet_dist)"
    # Test if it works
    if quartet_dist --help &> /dev/null || quartet_dist 2>&1 | grep -q "quartet"; then
        echo -e "${GREEN}[SUCCESS]${NC} quartet_dist is functional"
    else
        echo -e "${YELLOW}[WARNING]${NC} quartet_dist found but may not be working properly"
    fi
else
    missing_deps+=("quartet_dist")
    echo -e "${RED}[ERROR]${NC} quartet_dist not found in PATH"
fi

if command -v triplet_dist &> /dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} triplet_dist found: $(which triplet_dist)"
    # Test if it works
    if triplet_dist --help &> /dev/null || triplet_dist 2>&1 | grep -q "triplet"; then
        echo -e "${GREEN}[SUCCESS]${NC} triplet_dist is functional"
    else
        echo -e "${YELLOW}[WARNING]${NC} triplet_dist found but may not be working properly"
    fi
else
    missing_deps+=("triplet_dist")
    echo -e "${RED}[ERROR]${NC} triplet_dist not found in PATH"
fi

# Check booster (expect booster_linux64 on Linux systems)
BOOSTER_NAME="booster_linux64"
if [[ "$(uname)" == "Darwin" ]]; then
    BOOSTER_NAME="booster_macos64"
fi

if command -v "$BOOSTER_NAME" &> /dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} $BOOSTER_NAME found: $(which $BOOSTER_NAME)"
    # Test if it works
    if "$BOOSTER_NAME" --help &> /dev/null || "$BOOSTER_NAME" --version &> /dev/null; then
        echo -e "${GREEN}[SUCCESS]${NC} $BOOSTER_NAME is functional"
    else
        echo -e "${YELLOW}[WARNING]${NC} $BOOSTER_NAME found but may not be working properly"
    fi
elif command -v booster &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Found 'booster' but expected '$BOOSTER_NAME'"
    echo -e "${YELLOW}[WARNING]${NC} booster found at: $(which booster)"
    echo -e "${YELLOW}[WARNING]${NC} Code expects '$BOOSTER_NAME' - you may need to create a symlink"
else
    missing_deps+=("$BOOSTER_NAME")
    echo -e "${RED}[ERROR]${NC} $BOOSTER_NAME not found in PATH"
fi

# Check SCONCE2 (used in Snakemake workflow)
echo ""
echo -e "${BLUE}[INFO]${NC} Checking optional binaries for workflows..."

SCONCE2_PATH="$PROJECT_ROOT/external/sconce2/sconce2"
if [ -f "$SCONCE2_PATH" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} SCONCE2 found: $SCONCE2_PATH"
elif command -v sconce2 &> /dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} SCONCE2 found in PATH: $(which sconce2)"
else
    missing_optional+=("SCONCE2")
    echo -e "${YELLOW}[WARNING]${NC} SCONCE2 not found (needed for Snakemake workflows)"
fi

# Summary
echo ""
echo "================================================"
echo "Dependency Check Summary"
echo "================================================"

if [ ${#missing_deps[@]} -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} All required dependencies are available!"
    
    if [ ${#missing_optional[@]} -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} All optional dependencies are also available!"
        echo -e "${GREEN}[SUCCESS]${NC} Ready to run all SparseRNJ experiments and workflows"
    else
        echo -e "${YELLOW}[INFO]${NC} Missing optional dependencies: ${missing_optional[*]}"
        echo -e "${YELLOW}[INFO]${NC} Basic experiments will work, but some workflows may fail"
    fi
    
    exit 0
else
    echo -e "${RED}[ERROR]${NC} Missing required dependencies: ${missing_deps[*]}"
    echo ""
    echo -e "${BLUE}[INFO]${NC} Required external binaries must be installed manually:"
    echo ""
    
    if [[ " ${missing_deps[*]} " =~ " quartet_dist " ]] || [[ " ${missing_deps[*]} " =~ " triplet_dist " ]]; then
        echo -e "${BLUE}[INFO]${NC} tqDist (quartet_dist, triplet_dist):"
        echo -e "${BLUE}[INFO]${NC}   Download from: https://www.birc.au.dk/~cstorm/software/tqdist/"
        echo -e "${BLUE}[INFO]${NC}   Install and ensure quartet_dist and triplet_dist are in PATH"
    fi
    
    if [[ " ${missing_deps[*]} " =~ " booster_linux64 " ]] || [[ " ${missing_deps[*]} " =~ " booster_macos64 " ]]; then
        echo -e "${BLUE}[INFO]${NC} booster:"
        echo -e "${BLUE}[INFO]${NC}   Download from: https://github.com/evolbioinfo/booster"
        echo -e "${BLUE}[INFO]${NC}   Build and ensure $BOOSTER_NAME is in PATH"
        echo -e "${BLUE}[INFO]${NC}   (You may need to rename the executable to $BOOSTER_NAME)"
    fi
    
    if [ ${#missing_optional[@]} -gt 0 ]; then
        echo ""
        echo -e "${BLUE}[INFO]${NC} Optional dependencies:"
        if [[ " ${missing_optional[*]} " =~ " SCONCE2 " ]]; then
            echo -e "${BLUE}[INFO]${NC}   SCONCE2: Run ./scripts/setup_sconce2.sh"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}[INFO]${NC} After installing external binaries, run this check again to verify"
    
    exit 1
fi