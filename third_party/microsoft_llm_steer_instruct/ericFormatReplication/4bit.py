import os

# Identify the path to the utils.py you just showed me
# Based on your tree, it's likely in the utils/ folder
path = "../utils/model_utils.py" 
# If the file is named exactly 'utils.py' in your root, change this to "utils.py"

if not os.path.exists(path):
    # Fallback check
    path = "utils.py"

with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. We need to add 'torch' to the imports if it's missing (needed for bfloat16)
if "import torch" not in code:
    code = "import torch\n" + code

# 2. Update the AutoModelForCausalLM call to use 4-bit
# We add load_in_4bit, device_map (crucial for bitsandbytes), and bfloat16 (for your 4070/T4)
old_hf_load = "hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=hf_token, cache_dir=cache_dir)"
new_hf_load = """hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name, 
            token=hf_token, 
            cache_dir=cache_dir, 
            load_in_4bit=True, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )"""

if old_hf_load in code:
    code = code.replace(old_hf_load, new_hf_load)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)
    print("✅ utils.py patched for 4-bit. Your 8GB/15GB VRAM is now safe.")
else:
    print("❌ Could not find the loading line. Check if the file content matches exactly.")