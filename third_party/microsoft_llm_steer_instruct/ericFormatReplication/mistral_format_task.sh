#!/bin/bash
# 1. Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
export PYTHONPATH="$PYTHONPATH:$PARENT_DIR"

MODEL="mistralai/Mistral-7B-Instruct-v0.1"

python "UTF_8.py"

# 2. Call the Python Shuffler
python "$SCRIPT_DIR/prep_data.py"

# 3. Extraction (Point to the SIBLING folder)
python "$PARENT_DIR/format/compute_representations.py" \
    model_name="$MODEL" \
    use_data_subset=True \
    data_subset_ratio=0.1 \
    +batch_size=128

# 4. Search
# python "$PARENT_DIR/format/find_best_layer.py" \
#     model_name="$MODEL" \
#     representations_folder="subset_0.1" \
#     n_examples_per_instruction=1

# 5. Evaluate (This is the one that was failing with Errno 2)
python "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    include_instructions=false \
    max_generation_length=256 \