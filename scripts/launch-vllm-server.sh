#!/bin/bash
set -e

# Check if model path is provided as first argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [additional_vllm_args...]"
    echo "Example: $0 /path/to/model --port 30000 --dp 1 --tp 1"
    exit 1
fi

MODEL_PATH="$1"
shift  # Remove the first argument (model_path)

# All remaining arguments will be passed to vllm serve
VLLM_ARGS="$@"

# Default port for checking if server is running
DEFAULT_PORT=30000
PORT=$DEFAULT_PORT

# Extract port from arguments if provided
prev_arg=""
for arg in $VLLM_ARGS; do
    if [[ "$prev_arg" == "--port" ]]; then
        PORT="$arg"
        break
    fi
    prev_arg="$arg"
done

echo "Checking if vLLM server is already running on port ${PORT}..."
if curl -s http://localhost:${PORT}/v1/models > /dev/null; then
    echo "vLLM server is already running on port ${PORT}!"
else
    echo "Starting vLLM server on port ${PORT}..."
    # Map some common sglang flags to vllm flags if they were passed
    # This helps scripts that were using sglang flags directly
    MODIFIED_ARGS=""
    for arg in ${VLLM_ARGS}; do
        case $arg in
            --dp)
                MODIFIED_ARGS="${MODIFIED_ARGS} --data-parallel-size"
                ;;
            --tp)
                MODIFIED_ARGS="${MODIFIED_ARGS} --tensor-parallel-size"
                ;;
            --mem-fraction-static)
                MODIFIED_ARGS="${MODIFIED_ARGS} --gpu-memory-utilization"
                ;;
            --is-embedding)
                # Map to task-type for reward/embedding models if using vllm >= 0.6.0
                MODIFIED_ARGS="${MODIFIED_ARGS} --task-type embedding"
                ;;
            *)
                MODIFIED_ARGS="${MODIFIED_ARGS} $arg"
                ;;
        esac
    done

    vllm serve "${MODEL_PATH}" --host 0.0.0.0 ${MODIFIED_ARGS} 2>&1 | tee vllm-$(basename ${MODEL_PATH}).log &
    
    echo "Waiting for vLLM server to be ready..."
    while ! curl -s http://localhost:${PORT}/v1/models > /dev/null; do
        sleep 10
    done
    echo "vLLM server is ready!"
fi

