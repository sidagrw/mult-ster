import os

root = "/root/mult-ster/third_party/microsoft_llm_steer_instruct"

# 1. Fix the Loader (model_utils.py)
utils_path = os.path.join(root, "utils", "model_utils.py")
if os.path.exists(utils_path):
    with open(utils_path, 'r') as f:
        code = f.read()
    # Ensure device_map is passed but device is NOT forced later
    code = code.replace('load_in_4bit=True,', 'load_in_4bit=True, device_map="auto",')
    with open(utils_path, 'w') as f:
        f.write(code)
    print("✅ utils/model_utils.py updated.")

# 2. Fix the Scripts (evaluate.py and find_best_layer_fast.py)
scripts = [os.path.join(root, "format", "evaluate.py"), 
           os.path.join(root, "format", "find_best_layer_fast.py")]

for script in scripts:
    if os.path.exists(script):
        with open(script, 'r') as f:
            lines = f.readlines()
        with open(script, 'w') as f:
            for line in lines:
                # We wrap the .to(device) calls in a safety check
                if "model.to(device)" in line or "model.to(args.device)" in line:
                    indent = line[:line.find("model.to")]
                    f.write(f"{indent}if not getattr(model, 'is_quantized', False) and not hasattr(model, 'hf_device_map'):\n")
                    f.write(f"{indent}    {line.strip()}\n")
                else:
                    f.write(line)
        print(f"✅ {os.path.basename(script)} patched for 4-bit safety.")