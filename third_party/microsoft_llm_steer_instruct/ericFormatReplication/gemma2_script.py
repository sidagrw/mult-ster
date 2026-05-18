import os

# Path to the registry file in your 3GB unzipped env
tl_registry_path = "/content/env_core/content/temp_packages/transformer_lens/loading_from_pretrained.py"

if os.path.exists(tl_registry_path):
    with open(tl_registry_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    gemma2_name = "'google/gemma-2-2b-it'"
    if gemma2_name not in content:
        # Shove Gemma 2 into the official list
        content = content.replace("]", f", {gemma2_name}]")
        with open(tl_registry_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Gemma 2 officially added to the local registry.")
    else:
        print("💡 Gemma 2 already exists.")