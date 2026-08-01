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
SETTING_SUFFIX="no_instr"   # mirrors what the .py scripts derive from INCLUDE_INSTR
USE_PERPLEXITY=false  # whether precompute_ivs.py accounts for perplexity when picking best layers

# Set FORCE_RESWEEP=1 before running this script to bypass all skip/restore
# logic below and redo the sweep/perplexity steps even if output already
# exists locally or on Drive (e.g. after fixing a bug in find_best_layer.py).
FORCE_RESWEEP="${FORCE_RESWEEP:-0}"

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
# 3. Per-instruction layer sweep. Three-tier check:
#    (a) already local -> skip entirely
#    (b) not local, but exists on Drive -> restore, then skip
#    (c) neither -> actually run the expensive sweep
#    FORCE_RESWEEP=1 bypasses all three and always reruns.
# --------------------------------------------------------------------
LAYER_SWEEP_DIR="$PARENT_DIR/format/layer_search_out/$MODEL/n_examples${N_EXAMPLES}_seed${SEED}_mult_rs"
LAYER_SWEEP_FILE="$LAYER_SWEEP_DIR/out_${SETTING_SUFFIX}.jsonl"
DRIVE_LAYER_SWEEP_BACKUP="$DRIVE_BACKUP_DIR/${MODEL}_layer_search_out"
DRIVE_LAYER_SWEEP_FILE="$DRIVE_LAYER_SWEEP_BACKUP/$MODEL/n_examples${N_EXAMPLES}_seed${SEED}_mult_rs/out_${SETTING_SUFFIX}.jsonl"

if [ ! -f "$LAYER_SWEEP_FILE" ] && [ "$FORCE_RESWEEP" != "1" ] && [ -f "$DRIVE_LAYER_SWEEP_FILE" ]; then
    echo ">>> Local layer sweep output missing, but found the exact file on Drive at $DRIVE_LAYER_SWEEP_FILE"
    echo ">>> Restoring from Drive instead of recomputing"
    mkdir -p "$PARENT_DIR/format/layer_search_out"
    cp -r "$DRIVE_LAYER_SWEEP_BACKUP"/* "$PARENT_DIR/format/layer_search_out/"
fi

if [ -f "$LAYER_SWEEP_FILE" ] && [ "$FORCE_RESWEEP" != "1" ]; then
    echo ">>> Layer sweep output present at $LAYER_SWEEP_FILE (local or just restored)"
    echo ">>> Skipping find_best_layer.py (set FORCE_RESWEEP=1 to redo it anyway)"
else
    echo ">>> Running find_best_layer.py for $MODEL"
    python3 "$PARENT_DIR/format/find_best_layer.py" \
        model_name="$MODEL" \
        representations_folder="all" \
        n_examples_per_instruction=$N_EXAMPLES \
        seed=$SEED \
        include_instructions=$INCLUDE_INSTR \
        steering="mult_rs" \
        +mult_steering_weight=0.5

    # --- Back up immediately, right here, before step 4 even starts ---
    echo ">>> Backing up layer_search_out to Drive"
    mkdir -p "$DRIVE_LAYER_SWEEP_BACKUP"
    cp -r "$PARENT_DIR/format/layer_search_out"/* "$DRIVE_LAYER_SWEEP_BACKUP/"
    echo ">>> Backup done, safe to continue"
fi

# --------------------------------------------------------------------
# 4. Add GPT-2 perplexity to the sweep results. Same three-tier check
#    as step 3: local -> skip, Drive-only -> restore then skip,
#    neither -> actually run it.
# --------------------------------------------------------------------
PERPLEXITY_DIR="${LAYER_SWEEP_DIR}_with_perplexity"
PERPLEXITY_FILE="$PERPLEXITY_DIR/out_${SETTING_SUFFIX}.jsonl"
DRIVE_PERPLEXITY_BACKUP="$DRIVE_BACKUP_DIR/${MODEL}_layer_search_out_with_perplexity"
DRIVE_PERPLEXITY_FILE="$DRIVE_PERPLEXITY_BACKUP/n_examples${N_EXAMPLES}_seed${SEED}_mult_rs_with_perplexity/out_${SETTING_SUFFIX}.jsonl"

if [ ! -f "$PERPLEXITY_FILE" ] && [ "$FORCE_RESWEEP" != "1" ] && [ -f "$DRIVE_PERPLEXITY_FILE" ]; then
    echo ">>> Local perplexity output missing, but found the exact file on Drive at $DRIVE_PERPLEXITY_FILE"
    echo ">>> Restoring from Drive instead of recomputing"
    # The backup's top level IS the "..._with_perplexity" folder itself
    # (that's what compute_response_perplexity.py's own glob captured when
    # backing it up), so restore into the PARENT of PERPLEXITY_DIR, not
    # PERPLEXITY_DIR itself, or you'd get a double-nested folder.
    mkdir -p "$(dirname "$PERPLEXITY_DIR")"
    cp -r "$DRIVE_PERPLEXITY_BACKUP"/* "$(dirname "$PERPLEXITY_DIR")/"
fi

if [ -f "$PERPLEXITY_FILE" ] && [ "$FORCE_RESWEEP" != "1" ]; then
    echo ">>> Perplexity-augmented output present at $PERPLEXITY_FILE (local or just restored)"
    echo ">>> Skipping compute_response_perplexity.py (set FORCE_RESWEEP=1 to redo it anyway)"
else
    echo ">>> Running compute_response_perplexity.py for $MODEL"
    python3 "$PARENT_DIR/format/compute_response_perplexity.py" \
        model_name="$MODEL" \
        n_examples=$N_EXAMPLES \
        seed=$SEED \
        include_instructions=$INCLUDE_INSTR \
        +steering="mult_rs"

    mkdir -p "$DRIVE_PERPLEXITY_BACKUP"
    cp -r "$PARENT_DIR/format/layer_search_out"/*/*_with_perplexity "$DRIVE_PERPLEXITY_BACKUP/"
fi

# --------------------------------------------------------------------
# 5. Precompute instruction vectors / task matrices. Cheap, not skipped
#    -- rerun freely whenever you tweak preplexity_threshold or
#    USE_PERPLEXITY without redoing steps 3-4.
# --------------------------------------------------------------------
echo ">>> Running precompute_ivs.py for $MODEL"
python3 "$PARENT_DIR/format/precompute_ivs.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    n_examples=$N_EXAMPLES \
    seed=$SEED \
    include_instructions=$INCLUDE_INSTR \
    use_perplexity=$USE_PERPLEXITY \
    +steering="mult_rs"

# --- Back up the resulting pre_computed_ivs h5 too, same pattern ---
IVS_BACKUP_DIR="$DRIVE_BACKUP_DIR/${MODEL}_precomputed_ivs"
mkdir -p "$IVS_BACKUP_DIR"
cp "$REPS_DIR"/pre_computed_ivs_*.h5 "$IVS_BACKUP_DIR/" 2>/dev/null || true
echo ">>> Backed up precomputed IVs to $IVS_BACKUP_DIR"

# --------------------------------------------------------------------
# 6. Evaluate -- source_layer_idx=-1 tells evaluate.py to pull the
#    per-instruction optimal layer from the file just built in step 5.
#    add_vector / adjust_rs / none left commented, matching your last
#    edit -- uncomment if/when you want the full four-way comparison.
# --------------------------------------------------------------------
# echo ">>> Running evaluate.py: baseline (no steering)"
# python3 "$PARENT_DIR/format/evaluate.py" \
#     model_name="$MODEL" \
#     representations_folder="all" \
#     include_instructions=$INCLUDE_INSTR \
#     steering="none" \
#     source_layer_idx=-1

# echo ">>> Running evaluate.py: add_vector"
# python3 "$PARENT_DIR/format/evaluate.py" \
#     model_name="$MODEL" \
#     representations_folder="all" \
#     include_instructions=$INCLUDE_INSTR \
#     steering="add_vector" \
#     source_layer_idx=-1 \
#     steering_weight=2.0

# echo ">>> Running evaluate.py: adjust_rs"
# python3 "$PARENT_DIR/format/evaluate.py" \
#     model_name="$MODEL" \
#     representations_folder="all" \
#     include_instructions=$INCLUDE_INSTR \
#     steering="adjust_rs" \
#     source_layer_idx=-1

echo ">>> Running evaluate.py: mult_rs"
python3 "$PARENT_DIR/format/evaluate.py" \
    model_name="$MODEL" \
    representations_folder="all" \
    include_instructions=$INCLUDE_INSTR \
    steering="mult_rs" \
    source_layer_idx=-1 \
    +mult_steering_weight=0.5 \
    user_perplexity=$USE_PERPLEXITY

echo ">>> Done. Results under $PARENT_DIR/format/out/$MODEL/"