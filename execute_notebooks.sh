#!/bin/bash

# =============================================================================
# Notebook Execution Script for IndicForgeAI Translation Evaluation
# =============================================================================
# This script programmatically executes all Jupyter notebooks in the
# task1_translation_evaluation directory using jupyter nbconvert.
# 
# Prerequisites:
#   - jupyter >= 1.0.0
#   - nbconvert >= 6.0.0
#   - All notebook dependencies (transformers, torch, etc.)
#
# Usage:
#   ./execute_notebooks.sh
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define the base directory for notebooks
NOTEBOOK_BASE_DIR="/workspace/indic-translation-asr-project/task1_translation_evaluation"

# Define the notebooks to execute
NOTEBOOKS=(
    "part_a_batch_translation/part_a_translation_evaluation.ipynb"
    "part_b_token_analysis/part_b_token_eda.ipynb"
    "part_c_indic_token_behavior/part_c_indic_token_analysis.ipynb"
)

# Print header
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}  IndicForgeAI - Notebook Execution Script${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo -e "${RED}Error: jupyter command not found.${NC}"
    echo "Please install jupyter: pip install jupyter nbconvert"
    exit 1
fi

echo -e "${GREEN}✓ jupyter is installed:${NC} $(jupyter --version 2>&1 | head -1)"
echo ""

# Change to the notebook base directory
cd "$NOTEBOOK_BASE_DIR" || exit 1

echo -e "${YELLOW}Working directory: ${NC}$(pwd)"
echo ""

# Track execution statistics
TOTAL_NOTEBOOKS=${#NOTEBOOKS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0
START_TIME=$(date +%s)

# Execute each notebook
for i in "${!NOTEBOOKS[@]}"; do
    NOTEBOOK="${NOTEBOOKS[$i]}"
    NOTEBOOK_NUM=$((i + 1))
    
    echo -e "${BLUE}-----------------------------------------------------------------${NC}"
    echo -e "${BLUE}[${NOTEBOOK_NUM}/${TOTAL_NOTEBOOKS}] Executing: ${NOTEBOOK}${NC}"
    echo -e "${BLUE}-----------------------------------------------------------------${NC}"
    
    # Check if notebook file exists
    if [ ! -f "$NOTEBOOK" ]; then
        echo -e "${RED}✗ Error: Notebook not found: ${NOTEBOOK}${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi
    
    echo -e "${YELLOW}→ Starting execution at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    
    # Execute the notebook in-place with error handling
    # --to notebook: Convert to notebook format (preserves .ipynb)
    # --execute: Run all code cells
    # --inplace: Overwrite the original file with executed output
    # --allow-errors: Continue execution even if errors occur (for debugging)
    if jupyter nbconvert --to notebook --execute --inplace "$NOTEBOOK" 2>&1; then
        echo -e "${GREEN}✓ Successfully executed: ${NOTEBOOK}${NC}"
        echo -e "${YELLOW}→ Completed at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Failed to execute: ${NOTEBOOK}${NC}"
        echo -e "${YELLOW}→ Failed at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    echo ""
done

# Calculate total execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Print summary
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}  Execution Summary${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""
echo -e "Total Notebooks: ${TOTAL_NOTEBOOKS}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAILED_COUNT}${NC}"
echo ""
echo -e "Total Execution Time: ${MINUTES}m ${SECONDS}s"
echo ""

if [ $FAILED_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All notebooks executed successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some notebooks failed. Please check the logs above.${NC}"
    exit 1
fi
