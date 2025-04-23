#!/bin/bash

# Lovince AI Manager Script
# Created by: The Founder - Lovince ™
# Purpose: Automates the execution, management, and scaling of Lovince AI simulations
# Version: Final (April 22, 2025)
# License: Open for exploration, inspired by the cosmic vision of Lovince

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script variables
SCRIPT_NAME="Lovince AI Manager"
VERSION="1.0"
LOG_FILE="lovince_ai_manager.log"
PYTHON_SCRIPT="final_powerful.py"
RESULTS_DIR="lovince_results"
SEQUENCE_POWERS=(1 3 9 27 81) # Sequence: 3^(n-1)
NUM_RUNS=3 # Default number of runs per power
DEPENDENCIES=("numpy" "matplotlib" "pandas" "torch" "qiskit" "mpmath")

# Logging function
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    case "$level" in
        INFO) echo -e "${GREEN}[$level] $message${NC}";;
        WARNING) echo -e "${YELLOW}[$level] $message${NC}";;
        ERROR) echo -e "${RED}[$level] $message${NC}";;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check and install dependencies
check_dependencies() {
    log_message "INFO" "Checking dependencies..."
    if ! command_exists python3; then
        log_message "ERROR" "Python3 not found. Please install Python3."
        exit 1
    fi

    if ! command_exists pip3; then
        log_message "ERROR" "pip3 not found. Please install pip3."
        exit 1
    fi

    for dep in "${DEPENDENCIES[@]}"; do
        if ! pip3 show "$dep" >/dev/null 2>&1; then
            log_message "WARNING" "Dependency $dep not found. Installing..."
            if pip3 install "$dep" >> "$LOG_FILE" 2>&1; then
                log_message "INFO" "$dep installed successfully."
            else
                log_message "ERROR" "Failed to install $dep."
                exit 1
            fi
        else
            log_message "INFO" "$dep already installed."
        fi
    done
}

# Check if Python script exists
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_message "ERROR" "Python script $PYTHON_SCRIPT not found."
        exit 1
    fi
    log_message "INFO" "Python script $PYTHON_SCRIPT found."
}

# Create results directory
setup_results_dir() {
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir -p "$RESULTS_DIR"
        log_message "INFO" "Created results directory: $RESULTS_DIR"
    fi
}

# Run simulation for a specific power and run number
run_simulation() {
    local power="$1"
    local run="$2"
    local run_id="power_${power}_run_${run}"
    local output_dir="$RESULTS_DIR/$run_id"
    mkdir -p "$output_dir"

    log_message "INFO" "Starting simulation: Power=$power, Run=$run"
    # Run Python script with environment variable for power
    export LOVINCE_POWER="$power"
    if python3 "$PYTHON_SCRIPT" > "$output_dir/output.log" 2>&1; then
        log_message "INFO" "Simulation completed: Power=$power, Run=$run"
        # Move results
        mv lovince_ai_data_*.csv "$output_dir/" 2>/dev/null || true
        mv lovince_ai_results_*.png "$output_dir/" 2>/dev/null || true
        mv lovince_ai.log "$output_dir/lovince_ai.log" 2>/dev/null || true
        log_message "INFO" "Results moved to $output_dir"
    else
        log_message "ERROR" "Simulation failed: Power=$power, Run=$run. Check $output_dir/output.log"
        return 1
    fi
}

# Parallel execution of simulations
run_parallel_simulations() {
    log_message "INFO" "Starting parallel simulations..."
    local tasks=()
    
    for power in "${SEQUENCE_POWERS[@]}"; do
        for run in $(seq 1 "$NUM_RUNS"); do
            tasks+=("$power $run")
        done
    done

    # Run tasks in parallel (limit to number of CPU cores)
    local max_jobs=$(nproc)
    local running_jobs=0
    for task in "${tasks[@]}"; do
        power=$(echo "$task" | awk '{print $1}')
        run=$(echo "$task" | awk '{print $2}')
        run_simulation "$power" "$run" &
        ((running_jobs++))
        
        if [ "$running_jobs" -ge "$max_jobs" ]; then
            wait -n
            ((running_jobs--))
        fi
    done
    wait
    log_message "INFO" "All simulations completed."
}

# Validate results
validate_results() {
    log_message "INFO" "Validating results..."
    local valid=true
    for power in "${SEQUENCE_POWERS[@]}"; do
        for run in $(seq 1 "$NUM_RUNS"); do
            local run_id="power_${power}_run_${run}"
            local output_dir="$RESULTS_DIR/$run_id"
            if [ -f "$output_dir/lovince_ai.log" ] && [ -f "$output_dir/output.log" ]; then
                if grep -q "Sequence powers validated successfully" "$output_dir/lovince_ai.log"; then
                    log_message "INFO" "Validation passed for $run_id"
                else
                    log_message "WARNING" "Validation failed for $run_id"
                    valid=false
                fi
            else
                log_message "ERROR" "Missing results for $run_id"
                valid=false
            fi
        done
    done
    if [ "$valid" = true ]; then
        log_message "INFO" "All results validated successfully."
    else
        log_message "WARNING" "Some validations failed. Check $LOG_FILE for details."
    fi
}

# Display usage
usage() {
    echo "Usage: $0 [-r RUNS] [-h]"
    echo "  -r RUNS  Number of runs per power (default: $NUM_RUNS)"
    echo "  -h       Display this help message"
    echo "Example: $0 -r 5"
    exit 1
}

# Parse command-line arguments
while getopts "r:h" opt; do
    case "$opt" in
        r) NUM_RUNS="$OPTARG";;
        h) usage;;
        *) usage;;
    esac
done

# Main function
main() {
    log_message "INFO" "Starting $SCRIPT_NAME v$VERSION by The Founder - Lovince ™"
    log_message "INFO" "Reality Check: 99 + π/π = 100% real"

    # Setup
    check_dependencies
    check_python_script
    setup_results_dir

    # Run simulations
    run_parallel_simulations

    # Validate results
    validate_results

    log_message "INFO" "All tasks completed. Results stored in $RESULTS_DIR"
    log_message "INFO" "Thank you for exploring the Lovince AI universe! 🚀"
}

# Trap errors
trap 'log_message "ERROR" "Script terminated unexpectedly"; exit 1' ERR

# Run main
main

# Reset trap
trap - ERR