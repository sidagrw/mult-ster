@echo off
echo 🚀 Starting Replication...

:: 1. Activate your Conda environment
:: Note: You might need to use 'call' before conda
call conda activate mult_steer

cd /d "C:\Users\ryany\Algoverse25-26Spring\mult-ster\third_party\microsoft_llm_steer_instruct"

:: 2. Run Compute
python format/compute_representations.py ^
    model_name="Qwen/Qwen1.5-1.8B-Chat" ^
    use_data_subset=True ^
    data_subset_ratio=0.1 ^
    +batch_size=128

:: 3. Run Search
@REM python format/find_best_layer.py ^
@REM     model_name="Qwen/Qwen1.5-1.8B-Chat" ^
@REM     representations_folder="subset_0.1" ^
@REM     n_examples_per_instruction=1

:: 4. Run Evaluate
python format/evaluate.py ^
    model_name="Qwen/Qwen1.5-1.8B-Chat" ^
    include_instructions=false

echo ✅ DONE.

cd /d "C:\Users\ryany\Algoverse25-26Spring\mult-ster\third_party\microsoft_llm_steer_instruct\ericFormatReplication"
pause