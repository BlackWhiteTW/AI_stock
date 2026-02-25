import os
import shutil
from pathlib import Path

def standardize_path(base_dir, is_file=True):
    path = Path(base_dir)
    if not path.exists():
        print(f"Path does not exist: {base_dir}")
        return

    items = list(path.glob("*.pth")) if is_file else [d for d in path.iterdir() if d.is_dir()]
    
    mapping = {}
    for item in items:
        name = item.name.replace(".pth", "") if is_file else item.name
        code = name.split(".")[0]
        if not code.isdigit():
            continue
            
        if code not in mapping:
            mapping[code] = []
        mapping[code].append(item)

    for code, files in mapping.items():
        target_name = f"{code}.TW.pth" if is_file else f"{code}.TW"
        target_path = path / target_name
        
        if len(files) == 1:
            current_file = files[0]
            if current_file.name != target_name:
                print(f"Renaming: {current_file.name} -> {target_name}")
                current_file.rename(target_path)
        else:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            newest = files[0]
            others = files[1:]
            
            print(f"Merging {code}: Keeping newest ({newest.name}), deleting others.")
            
            temp_path = path / f"{target_name}.tmp"
            if is_file:
                shutil.copy2(newest, temp_path)
            else:
                shutil.copytree(newest, temp_path)

            for f in files:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            
            temp_path.rename(target_path)

if __name__ == "__main__":
    print("--- Standardizing model files ---")
    standardize_path("data/saved_models", is_file=True)
    
    print("\n--- Standardizing financial data directories ---")
    standardize_path("data/financial_data", is_file=False)
    
    print("\nDone!")
