#!/bin/bash
# 1. Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
export PYTHONPATH="$PYTHONPATH:$PARENT_DIR"

mkdir -p /root/mult-ster/third_party/microsoft_llm_steer_instruct/format/representations
unzip -q /content/drive/MyDrive/reps.zip -d /root/mult-ster/third_party/microsoft_llm_steer_instruct/format/representations
MODEL="google/gemma-2-2b-it"
BEST_LAYER=20

python "UTF_8.py"

# 2. Call the Python Shuffler
python3 "$SCRIPT_DIR/prep_data.py"

# 3. Extraction (Point to the SIBLING folder)
# python "$PARENT_DIR/format/compute_representations.py" \
#     model_name="$MODEL" \
#     use_data_subset=True \
#     data_subset_ratio=0.1 \

# 4. Search
# python3 "$PARENT_DIR/format/find_best_layer.py" \
#     model_name="$MODEL" \
#     representations_folder="subset_0.1" \
#     n_examples_per_instruction=5

python3 format/precompute_ivs.py \
        model_name="$MODEL" \
        representations_folder="subset_0.1" \
        specific_layer=$BEST_LAYER

# 5. Evaluate (This is the one that was failing with Errno 2)
python3 "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    representations_folder="subset_0.1" \ 
    include_instructions=false \
    source_layer_idx=$BEST_LAYER \
    max_generation_length=128 \
    steering_weight=1