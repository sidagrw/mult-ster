#!/bin/bash
set -e  # stop on first error instead of silently continuing into a broken later stage

# --------------------------------------------------------------------
# 0. Setup paths
# --------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
export PYTHONPATH="$PYTHONPATH:$PARENT_DIR"

MODEL="phi-3"
DRIVE_BACKUP_DIR="/content/drive/Shareddrives/Eric/LLM_STEER_BACKUP"
REPS_DIR="$PARENT_DIR/format/representations/$MODEL/all"
REPS_ZIP="$DRIVE_BACKUP_DIR/${MODEL}_reps.zip"

# n_examples / seed / include_instructions must be identical across
# find_best_layer.py, compute_response_perplexity.py, and precompute_ivs.py,
# since they're encoded into the folder name each one reads/writes.
N_EXAMPLES=8
SEED=42
INCLUDE_INSTR=false   # matches the no_instr evaluate.py calls below

mkdir -p "$PARENT_DIR/format/representations"
mkdir -p "$DRIVE_BACKUP_DIR"

# --------------------------------------------------------------------
# 1. (OPTIONAL) Restore representations from a previous backup instead
#    of recomputing. Skipped automatically if no backup exists yet --
#    safe to leave in for a first run.
# --------------------------------------------------------------------
if [ -f "$REPS_ZIP" ]; then
    echo ">>> Found existing backup at $REPS_ZIP, restoring instead of recomputing."
    unzip -q -o "$REPS_ZIP" -d "$PARENT_DIR/format/representations"
else
    echo ">>> No existing backup found at $REPS_ZIP -- will compute representations from scratch."
fi

# --------------------------------------------------------------------
# 2. Extraction (skipped if representations were just restored above)
# --------------------------------------------------------------------
if [ ! -d "$REPS_DIR" ] || [ -z "$(ls -A "$REPS_DIR" 2>/dev/null)" ]; then
    echo ">>> Running compute_representations.py for $MODEL"
    python3 "$PARENT_DIR/format/compute_representations.py" \
        model_name="$MODEL" \
        use_data_subset=false

    # --- Zip immediately after extraction, purely in bash ---
    echo ">>> Zipping representations to $REPS_ZIP"
    (cd "$PARENT_DIR/format/representations" && zip -rq "$REPS_ZIP" "$MODEL")
    echo ">>> Backup saved: $REPS_ZIP"
else
    echo ">>> Representations already present at $REPS_DIR, skipping extraction."
fi

# --------------------------------------------------------------------
# 3. Per-instruction layer sweep (cheap: n_examples_per_instruction=$N_EXAMPLES,
#    stride-2 layer range -- NOT the full 20+ hour sweep)
# --------------------------------------------------------------------
echo ">>> Running find_best_layer.py for $MODEL"
python3 "$PARENT_DIR/format/find_best_layer.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    n_examples_per_instruction=$N_EXAMPLES \
    seed=$SEED \
    include_instructions=$INCLUDE_INSTR \
    steering="mult_rs" \
    +mult_steering_weight=0.5

# --------------------------------------------------------------------
# 4. Add GPT-2 perplexity to the sweep results (the missing middle step --
#    precompute_ivs.py's use_perplexity=True path requires this file to exist)
# --------------------------------------------------------------------
echo ">>> Running compute_response_perplexity.py for $MODEL"
python3 "$PARENT_DIR/format/compute_response_perplexity.py" \
    model_name="$MODEL" \
    n_examples=$N_EXAMPLES \
    seed=$SEED \
    include_instructions=$INCLUDE_INSTR \
    +steering="mult_rs"

# --------------------------------------------------------------------
# 5. Precompute instruction vectors / task matrices at the REAL
#    per-instruction optimal layer (perplexity-gated, argmax accuracy,
#    ties -> earliest layer -- exactly Appendix E). No specific_layer
#    override, so its own internal optimal-layer selection is used.
# --------------------------------------------------------------------
echo ">>> Running precompute_ivs.py for $MODEL"
python3 "$PARENT_DIR/format/precompute_ivs.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    n_examples=$N_EXAMPLES \
    seed=$SEED \
    include_instructions=$INCLUDE_INSTR \
    use_perplexity=true \
    +steering="mult_rs"

# --- Back up the resulting pre_computed_ivs h5 too, same pattern ---
IVS_BACKUP_DIR="$DRIVE_BACKUP_DIR/${MODEL}_precomputed_ivs"
mkdir -p "$IVS_BACKUP_DIR"
cp "$REPS_DIR"/pre_computed_ivs_*.h5 "$IVS_BACKUP_DIR/" 2>/dev/null || true
echo ">>> Backed up precomputed IVs to $IVS_BACKUP_DIR"

# --------------------------------------------------------------------
# 6. Evaluate -- source_layer_idx=-1 tells evaluate.py to pull the
#    per-instruction optimal layer from the file just built in step 5,
#    instead of one blanket layer for every instruction.
# --------------------------------------------------------------------
echo ">>> Running evaluate.py: baseline (no steering)"
python3 "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    include_instructions=$INCLUDE_INSTR \
    steering="none" \
    source_layer_idx=-1

echo ">>> Running evaluate.py: add_vector"
python3 "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    include_instructions=$INCLUDE_INSTR \
    steering="add_vector" \
    source_layer_idx=-1 \
    steering_weight=2.0

echo ">>> Running evaluate.py: adjust_rs"
python3 "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    include_instructions=$INCLUDE_INSTR \
    steering="adjust_rs" \
    source_layer_idx=-1

echo ">>> Running evaluate.py: mult_rs"
python3 "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    include_instructions=$INCLUDE_INSTR \
    steering="mult_rs" \
    source_layer_idx=-1 \
    +mult_steering_weight=0.5

echo ">>> Done. Results under $PARENT_DIR/format/out/$MODEL/"