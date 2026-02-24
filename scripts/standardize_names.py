
import os
import shutil
from pathlib import Path

def standardize_path(base_dir, is_file=True):
    path = Path(base_dir)
    if not path.exists():
        print(f"路徑不存在: {base_dir}")
        return

    # 取得所有項目
    items = list(path.glob("*.pth")) if is_file else [d for d in path.iterdir() if d.is_dir()]
    
    # 建立對映表 {股票代碼: [檔案路徑, ...]}
    mapping = {}
    for item in items:
        name = item.name.replace(".pth", "") if is_file else item.name
        
        # 提取純數字部分 (例如 從 2330.TW 或 2330 提取 2330)
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
            # 只有一個檔案，若不是 .TW 則更名
            current_file = files[0]
            if current_file.name != target_name:
                print(f"更名: {current_file.name} -> {target_name}")
                current_file.rename(target_path)
        else:
            # 有多個檔案 (例如 2330.pth 和 2330.TW.pth)
            # 依最後修改時間排序，保留最新的
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            newest = files[0]
            others = files[1:]
            
            print(f"合併 {code}: 保留最新 ({newest.name})，刪除其餘 {len(others)} 個檔案")
            
            # 如果最新的檔案名稱不正確，先更名（先避開名稱衝突）
            temp_path = path / f"{target_name}.tmp"
            if newest.name != target_name:
                shutil.copy2(newest, temp_path)
            else:
                shutil.copy2(newest, temp_path)

            # 刪除所有舊的
            for f in files:
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
            
            # 將暫存的最新檔案更名為目標名稱
            temp_path.rename(target_path)

if __name__ == "__main__":
    print("--- 開始標準化模型檔案 ---")
    standardize_path("data/saved_models", is_file=True)
    
    print("
--- 開始標準化財務數據目錄 ---")
    standardize_path("data/financial_data", is_file=False)
    
    print("
完成！")
