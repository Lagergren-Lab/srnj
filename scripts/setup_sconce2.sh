#!/bin/bash

# Simple SCONCE2 repository download for SparseRNJ workflows
# Downloads SCONCE2 source code to external/sconce2/ 
# User must build SCONCE2 manually if needed

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="$PROJECT_ROOT/external/sconce2"

echo "Downloading SCONCE2 repository to: $INSTALL_DIR"

# Create directory
mkdir -p "$PROJECT_ROOT/external"

# Clone or update repository
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing SCONCE2 repository..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning SCONCE2 repository..."
    git clone https://github.com/NielsenBerkeleyLab/sconce2.git "$INSTALL_DIR"
fi

echo "SCONCE2 repository downloaded successfully"
echo ""
echo "To build SCONCE2 (requires boost, gsl libraries):"
echo "  cd $INSTALL_DIR"
echo "  make"
echo ""
echo "Or install SCONCE2 system-wide and ensure 'sconce2' is in PATH"