#!/bin/bash
#
# Unified SO101 Robot Data Collection Launcher
#
# This script starts both DORA dataflow and CLI with proper ordering:
# 1. Cleans up stale ZeroMQ socket files
# 2. Starts DORA dataflow in background
# 3. Waits for ZeroMQ sockets to be ready
# 4. Starts CLI in foreground
# 5. Handles cleanup on exit
#
# Usage:
#   bash scripts/run_so101.sh [options]
#
# Options are passed directly to the CLI (main.py)

set -e

# Configuration - Single unified environment
CONDA_ENV="${CONDA_ENV:-dorobot}"

# NPU Configuration - set USE_NPU=1 to enable Ascend NPU support
USE_NPU="${USE_NPU:-0}"
ASCEND_TOOLKIT_PATH="${ASCEND_TOOLKIT_PATH:-/usr/local/Ascend/ascend-toolkit}"

# Cloud Offload Configuration - set CLOUD_OFFLOAD=1 to skip local video encoding
# When enabled, raw images are kept and uploaded to cloud for encoding/training
CLOUD_OFFLOAD="${CLOUD_OFFLOAD:-0}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DORA_DIR="$PROJECT_ROOT/operating_platform/robot/robots/so101_v1"
SOCKET_IMAGE="/tmp/dora-zeromq-so101-image"
SOCKET_JOINT="/tmp/dora-zeromq-so101-joint"
SOCKET_TIMEOUT="${SOCKET_TIMEOUT:-30}"  # seconds to wait for sockets
DORA_INIT_DELAY="${DORA_INIT_DELAY:-5}"  # seconds to wait after sockets ready for DORA to fully initialize
DORA_PID=""
DORA_GRAPH_NAME=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Initialize conda for this shell
init_conda() {
    # Find conda installation
    if [ -n "$CONDA_EXE" ]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif [ -d "/opt/conda" ]; then
        CONDA_BASE="/opt/conda"
    else
        log_error "Cannot find conda installation. Please ensure conda is installed."
        exit 1
    fi

    # Source conda.sh to enable conda activate
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        log_error "Cannot find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
        exit 1
    fi
}

# Activate the conda environment
activate_env() {
    local env_name="$1"

    # Check if environment exists
    if ! conda env list | grep -q "^${env_name} "; then
        log_error "Conda environment '$env_name' does not exist."
        log_error "Please run: bash scripts/setup_env.sh"
        exit 1
    fi

    conda activate "$env_name"
    log_info "Activated conda environment: $env_name"
}

# Source Ascend NPU environment if needed
setup_npu_env() {
    if [ "$USE_NPU" == "1" ]; then
        log_step "Setting up Ascend NPU environment..."

        local set_env_script="$ASCEND_TOOLKIT_PATH/set_env.sh"
        if [ -f "$set_env_script" ]; then
            source "$set_env_script"
            log_info "Sourced CANN environment from: $set_env_script"
        else
            log_warn "CANN set_env.sh not found at: $set_env_script"
            log_warn "NPU may not work correctly. Set ASCEND_TOOLKIT_PATH if needed."
        fi
    fi
}

# Cleanup function - called on exit
cleanup() {
    log_step "Cleaning up..."

    # Stop DORA dataflow if running
    if [ -n "$DORA_GRAPH_NAME" ]; then
        log_info "Stopping DORA dataflow: $DORA_GRAPH_NAME"
        dora stop "$DORA_GRAPH_NAME" 2>/dev/null || true
    fi

    # Kill DORA process if still running
    if [ -n "$DORA_PID" ] && kill -0 "$DORA_PID" 2>/dev/null; then
        log_info "Killing DORA process (PID: $DORA_PID)"
        kill "$DORA_PID" 2>/dev/null || true
        wait "$DORA_PID" 2>/dev/null || true
    fi

    # Clean up socket files
    if [ -S "$SOCKET_IMAGE" ]; then
        rm -f "$SOCKET_IMAGE"
    fi
    if [ -S "$SOCKET_JOINT" ]; then
        rm -f "$SOCKET_JOINT"
    fi

    log_info "Cleanup complete"
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Clean up stale socket files from previous runs
cleanup_stale_sockets() {
    log_step "Checking for stale socket files..."

    if [ -e "$SOCKET_IMAGE" ]; then
        log_warn "Removing stale socket: $SOCKET_IMAGE"
        rm -f "$SOCKET_IMAGE"
    fi

    if [ -e "$SOCKET_JOINT" ]; then
        log_warn "Removing stale socket: $SOCKET_JOINT"
        rm -f "$SOCKET_JOINT"
    fi
}

# Wait for ZeroMQ sockets to be created
wait_for_sockets() {
    log_step "Waiting for ZeroMQ sockets to be ready..."

    local elapsed=0
    local poll_interval=0.5

    while [ $elapsed -lt $SOCKET_TIMEOUT ]; do
        # Check if both socket files exist
        if [ -S "$SOCKET_IMAGE" ] && [ -S "$SOCKET_JOINT" ]; then
            log_info "ZeroMQ sockets ready!"
            return 0
        fi

        # Show progress
        printf "\r  Waiting... %ds / %ds" $elapsed $SOCKET_TIMEOUT

        sleep $poll_interval
        elapsed=$((elapsed + 1))
    done

    echo ""  # New line after progress
    log_error "Timeout waiting for ZeroMQ sockets after ${SOCKET_TIMEOUT}s"
    log_error "  Expected: $SOCKET_IMAGE"
    log_error "  Expected: $SOCKET_JOINT"
    return 1
}

# Start DORA dataflow
start_dora() {
    log_step "Starting DORA dataflow..."

    # Check if dora command is available
    if ! command -v dora &> /dev/null; then
        log_error "'dora' command not found. Please ensure dora-rs is installed in the '$CONDA_ENV' environment."
        exit 1
    fi

    # Check if dataflow file exists
    local dataflow_file="$DORA_DIR/dora_teleoperate_dataflow.yml"
    if [ ! -f "$dataflow_file" ]; then
        log_error "Dataflow file not found: $dataflow_file"
        exit 1
    fi

    # Start DORA in background
    cd "$DORA_DIR"

    log_info "Running: dora run dora_teleoperate_dataflow.yml"
    dora run dora_teleoperate_dataflow.yml &
    DORA_PID=$!

    # Give DORA a moment to initialize
    sleep 2

    # Check if DORA is still running
    if ! kill -0 "$DORA_PID" 2>/dev/null; then
        log_error "DORA failed to start"
        exit 1
    fi

    log_info "DORA started (PID: $DORA_PID)"

    # Try to get the graph name for cleaner shutdown
    DORA_GRAPH_NAME=$(dora list 2>/dev/null | grep -oP 'dora_teleoperate_dataflow[^\s]*' | head -1) || true

    cd "$PROJECT_ROOT"
}

# Start CLI
start_cli() {
    log_step "Starting CLI..."

    # Default parameters (can be overridden via command line)
    local repo_id="${REPO_ID:-so101-test}"
    local single_task="${SINGLE_TASK:-start and test so101 arm.}"

    log_info "Running main.py with parameters:"
    log_info "  repo_id: $repo_id"
    log_info "  single_task: $single_task"
    if [ "$CLOUD_OFFLOAD" == "1" ]; then
        log_info "  cloud_offload: enabled (skip local video encoding)"
    fi

    # Build command arguments
    local cmd_args=(
        --robot.type=so101
        --record.repo_id="$repo_id"
        --record.single_task="$single_task"
    )

    # Add cloud_offload if enabled
    if [ "$CLOUD_OFFLOAD" == "1" ]; then
        cmd_args+=(--record.cloud_offload=true)
    fi

    # Start CLI in foreground (blocks until exit)
    python "$PROJECT_ROOT/operating_platform/core/main.py" \
        "${cmd_args[@]}" \
        "$@"
}

# Print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "SO101 Robot Data Collection - Unified Launcher"
    echo ""
    echo "Environment Variables:"
    echo "  CONDA_ENV           Conda environment name (default: dorobot)"
    echo "  REPO_ID             Dataset repository ID (default: so101-test)"
    echo "  SINGLE_TASK         Task description (default: 'start and test so101 arm.')"
    echo "  USE_NPU             Set to 1 to enable Ascend NPU support (default: 0)"
    echo "  CLOUD_OFFLOAD       Set to 1 to skip local video encoding (default: 0)"
    echo "                      When enabled, raw images are kept for cloud upload/training"
    echo "  ASCEND_TOOLKIT_PATH Path to CANN toolkit (default: /usr/local/Ascend/ascend-toolkit)"
    echo "  DORA_INIT_DELAY     Seconds to wait for DORA to initialize (default: 5)"
    echo "  SOCKET_TIMEOUT      Seconds to wait for ZeroMQ sockets (default: 30)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  REPO_ID=my-dataset $0"
    echo "  $0 --record.repo_id=custom-dataset"
    echo ""
    echo "  # With Ascend NPU support:"
    echo "  USE_NPU=1 $0"
    echo ""
    echo "  # With cloud offload (skip local encoding, upload to cloud):"
    echo "  CLOUD_OFFLOAD=1 $0"
    echo ""
    echo "  # NPU + cloud offload (recommended for Orange Pi):"
    echo "  USE_NPU=1 CLOUD_OFFLOAD=1 $0"
    echo ""
    echo "  # With longer init delay (if timeout issues):"
    echo "  DORA_INIT_DELAY=10 $0"
    echo ""
    echo "Note: This script starts both DORA dataflow and CLI automatically."
    echo "      Press 'n' to save episode and start new one."
    echo "      Press 'e' to stop recording and exit (with cloud training if CLOUD_OFFLOAD=1)."
}

# Main entry point
main() {
    # Handle help flag
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        print_usage
        exit 0
    fi

    echo ""
    echo "=========================================="
    echo "  SO101 Robot Data Collection Launcher"
    echo "=========================================="
    echo ""

    # Step 0: Initialize and activate conda environment
    log_step "Initializing conda environment..."
    init_conda
    activate_env "$CONDA_ENV"

    # Step 0.5: Setup NPU environment if needed
    setup_npu_env

    # Step 1: Clean up stale sockets
    cleanup_stale_sockets

    # Step 2: Start DORA
    start_dora

    # Step 3: Wait for sockets
    if ! wait_for_sockets; then
        log_error "Failed to initialize. Check DORA logs for errors."
        exit 1
    fi

    # Step 3.5: Wait for DORA to fully initialize (cameras, arms, etc.)
    log_step "Waiting ${DORA_INIT_DELAY}s for DORA nodes to fully initialize..."
    for i in $(seq $DORA_INIT_DELAY -1 1); do
        printf "\r  Initializing... %ds remaining" $i
        sleep 1
    done
    echo ""

    echo ""
    log_info "All systems ready!"
    echo ""
    echo "=========================================="
    echo "  Controls:"
    echo "    'n' - Save episode and start new one"
    if [ "$CLOUD_OFFLOAD" == "1" ]; then
        echo "    'e' - Stop, upload to cloud, and train"
        echo ""
        echo "  Mode: Cloud Offload (skip local encoding)"
    else
        echo "    'e' - Stop recording and exit"
    fi
    echo "=========================================="
    echo ""

    # Step 4: Start CLI (blocks until exit)
    start_cli "$@"

    log_info "Recording session ended"
}

# Run main
main "$@"
