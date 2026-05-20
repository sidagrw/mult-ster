#!/bin/bash

# 1. PATH SETUP
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
export PYTHONPATH="$PARENT_DIR:$PYTHONPATH"

echo "🛠️ PHASE 0: Manual Environment Fix..."
# Install the exact versions to get out of Package Hell
mkdir -p /root/mult-ster/third_party/microsoft_llm_steer_instruct/format/representations
unzip -q /content/drive/MyDrive/reps.zip -d /root/mult-ster/third_party/microsoft_llm_steer_instruct/format/representations

# pip install -r requirements.txt
# pip install bitsandbytes

# Apply the UTF-8 and NLTK fixes
python3 "$SCRIPT_DIR/UTF_8.py"
python3 4bit.py

huggingface-cli login

MODELS=("google/gemma-2-2b-it" "mistralai/Mistral-7B-Instruct-v0.1" "microsoft/Phi-3-mini-4k-instruct")
STEERING_METHODS=("add_vector" "adjust_rs")

for MODEL in "${MODELS[@]}"; do
    echo "===================================================="
    echo "🎯 MODEL: $MODEL"
    echo "===================================================="

    # Registry Injection for Gemma-2 / Phi-3
    python3 -c "import transformer_lens.loading_from_pretrained as l; l.OFFICIAL_MODEL_NAMES.append('$MODEL') if '$MODEL' not in l.OFFICIAL_MODEL_NAMES else None"

    # Step 1: Extraction (Skip if folder exists)
    REP_FOLDER="$PARENT_DIR/format/representations/$MODEL/subset_0.1"
    if [ -d "$REP_FOLDER" ] && [ "$(ls -A $REP_FOLDER)" ]; then
        echo "✅ Reps found. Skipping extraction."
    else
        echo "🚀 Extracting activations..."
        python3 "$PARENT_DIR/format/compute_representations.py" \
            model_name="$MODEL" \
            use_data_subset=True \
            data_subset_ratio=0.1 \
            +batch_size=1
    fi

    # python3 format/find_best_layer.py \
    #     model_name="$MODEL" \
    #     representations_folder="subset_0.1" \
    #     n_examples_per_instruction=1 \
    #     max_generation_length=32 \
    #     output_path="layer_search_out/$MODEL/results_folder" \
    #     +batch_size=1

    # # 2. PRECOMPUTE
    # # Now that we patched the script, it will look in 'results_folder'
    python3 ericFormatReplication/layer14.py

    python3 format/precompute_ivs.py \
        model_name="$MODEL" \
        representations_folder="subset_0.1" \
        +batch_size=1 \
        +specific_layer=14

    # Step 2: Sweep through Steering Methods
    for METHOD in "${STEERING_METHODS[@]}"; do
        echo "📊 EVALUATING: Method=$METHOD"
        python3 "$PARENT_DIR/format/evaluate.py" \
            model_name="$MODEL" \
            include_instructions=false \
            steering="$METHOD" \
            representations_folder="subset_0.1" \
            max_generation_length=256 \
            source_layer_idx=14 \
            use_perplexity=false \
            +batch_size=1
        echo "💾 Syncing results for $MODEL ($METHOD) to Drive..."
        mkdir -p "/content/drive/MyDrive/ICLR_2026_RESULTS/$MODEL"
        cp -r "$PARENT_DIR/format/out/"* "/content/drive/MyDrive/ICLR_2026_RESULTS/"
    done
done

echo "🏁 ALL MODELS AND METHODS COMPLETE."