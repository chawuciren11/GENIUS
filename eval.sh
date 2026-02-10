#!/bin/bash

# 1. Basic environment configuration

BASE_DIR=""
PYTHON_EXE="python3"
SCRIPT_NAME="eval.py"

# 2. Evaluation parameter configuration

MAX_WORKERS=20
API_URL=""    # the whole url will be {API_URL}/v1beta/models/{API_MODEL_NAME}:generateContent/
API_KEY="sk-xx"
API_MODEL_NAME="gemini-3-pro-preview"

# 3. Dimension and model list configuration 

DIMENSIONS=("implicit_pattern" "symbolic_constraint" "visual_constraint" "prior_conflicting" "multi_semantic")
MODELS=("nanobanana2")

# 4. Execution logic (automatic loop)

TOTAL_DIMS=${#DIMENSIONS[@]}
TOTAL_MODELS=${#MODELS[@]}
TOTAL_TASKS=$((TOTAL_DIMS * TOTAL_MODELS))
CURRENT_TASK=1

echo ">>> Preparing to start automated evaluation tasks, total: $TOTAL_TASKS tasks."

for MODEL in "${MODELS[@]}"; do
    for DIM in "${DIMENSIONS[@]}"; do
        
        echo "--------------------------------------------------------"
        echo ">>> [Task $CURRENT_TASK/$TOTAL_TASKS] Evaluating..."
        echo ">>> Model: $MODEL"
        echo ">>> Dimension: $DIM"
        echo "--------------------------------------------------------"

        $PYTHON_EXE "$SCRIPT_NAME" \
            --base_path "$BASE_DIR" \
            --dimension "$DIM" \
            --model "$MODEL" \
            --max_workers "$MAX_WORKERS" \
            --url "$API_URL" \
            --api_key "$API_KEY" \
            --api_model_name "$API_MODEL_NAME" \
            --metric ""

        # Check execution status of each task
        if [ $? -eq 0 ]; then
            echo ">>> [SUCCESS] Task $CURRENT_TASK completed."
        else
            echo ">>> [FAILED] Task $CURRENT_TASK (Model: $MODEL, Dimension: $DIM) encountered an error."
           
        fi

        CURRENT_TASK=$((CURRENT_TASK + 1))
    done
done

echo "========================================================"
echo ">>> All evaluation tasks have finished execution!"
echo ">>> Results directory: $BASE_DIR/evaluation/"
echo "========================================================"