import os

# Identify the root of the project (one level up from your replication folder)
root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Files to fix
targets = [
    os.path.join(root, "format", "compute_representations.py"),
    os.path.join(root, "format", "find_best_layer.py"),
    os.path.join(root, "format", "evaluate.py")
]

def force_utf8(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Missing: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Locate common data loading lines and force encoding
    content = content.replace("with open(f'{project_dir}/{args.data_path}') as f:", 
                              "with open(f'{project_dir}/{args.data_path}', encoding='utf-8') as f:")
    
    content = content.replace("with open(out_path, 'w') as f:", 
                              "with open(out_path, 'w', encoding='utf-8') as f:")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Forced UTF-8 on {os.path.basename(file_path)}")

for t in targets:
    force_utf8(t)