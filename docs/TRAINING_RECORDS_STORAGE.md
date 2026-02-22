# 訓練記錄存儲系統升級說明

## 📊 新的存儲方式

### 變更內容
- **舊方式**: 單一 JSON 文件 (`data/training_history.json`)
- **新方式**: 獨立文件夾 + 日期編號文件 (`data/training_records/training_YYYYMMDD.json`)
- **同日多次**: 自動添加序號 (`training_YYYYMMDD_02.json`, `training_YYYYMMDD_03.json`)

### 優勢

#### ✅ 避免單一文件過大
每次訓練生成獨立文件，不會讓單個 JSON 文件無限增長

#### ✅ 便於按日期查找和管理
文件名使用日期格式，可快速定位特定日期的訓練記錄

#### ✅ 可選擇性刪除舊記錄
可以手動刪除特定日期的記錄，不影響其他記錄

#### ✅ 支援資料夾直接開啟
一鍵開啟記錄資料夾，瀏覽所有訓練記錄

---

## 📁 文件結構

```
data/
└── training_records/
    ├── training_20260208.json        # 2/8 第一次訓練
    ├── training_20260208_02.json     # 2/8 第二次訓練
    ├── training_20260207.json        # 2/7 訓練記錄
    └── training_20260206.json        # 2/6 訓練記錄
```

---

## 📄 記錄格式

### 個股模型訓練
```json
{
  "type": "stock_model",
  "ticker": "2330.TW",
  "stock_name": "台積電",
  "duration": 45.3,
  "final_loss": 0.002134,
  "best_loss": 0.001987,
  "data_size": 1250,
  "timestamp": "2026-02-08T14:30:25"
}
```

### 主模型訓練
```json
{
  "type": "base_model",
  "mode": "增強版",
  "enhanced": true,
  "duration": 1234.5,
  "timestamp": "2026-02-07T10:15:30",
  "result": "✅ 訓練完成！25 支標的，5000 天數據"
}
```

### 批量訓練（熱門股）
```json
{
  "type": "batch_popular",
  "total": 5,
  "success": 4,
  "failed": 1,
  "timestamp": "2026-02-06T16:45:12"
}
```

### 批量訓練（智能篩選）
```json
{
  "type": "batch_smart",
  "total": 1500,
  "success": 127,
  "skipped": 1350,
  "error": 23,
  "timestamp": "2026-02-05T09:20:45"
}
```

---

## 🔄 遷移舊記錄

如果您之前有訓練記錄，可以使用遷移工具：

```bash
python scripts/migrate_training_records.py
```

遷移工具會：
1. 讀取舊的 `data/training_history.json`
2. 將每條記錄轉換為獨立文件
3. 保存到 `data/training_records/` 文件夾
4. 詢問是否備份舊文件

---

## 🎯 UI 功能

在 **Tab 7 - AI 模型訓練中心 > 📊 訓練歷史與統計** 中：

### 📊 統計信息
- 總訓練次數
- 主模型訓練次數
- 個股訓練次數
- 批量訓練次數

### 💾 存儲信息
- 記錄文件數
- 總磁盤使用
- 平均文件大小
- 存儲位置路徑

### 📋 記錄表格
- 顯示最近 50 筆記錄
- 支援排序和搜尋
- 彩色圖標區分類型

### 🔧 管理功能
- 清除所有記錄（帶確認對話框）
- 一鍵開啟記錄資料夾
- 顯示記錄時間範圍

---

## 🛠️ 手動管理

### 查看記錄
```powershell
# 列出所有記錄
Get-ChildItem data\training_records\*.json

# 查看特定記錄
Get-Content data\training_records\training_20260208_143025.json
```

### 刪除舊記錄
```powershell
# 刪除 2026 年 1 月的記錄
Get-ChildItem data\training_records\training_202601*.json | Remove-Item

# 刪除特定日期
Remove-Item data\training_records\training_20260208*.json

# 只保留最近 30 天
Get-ChildItem data\training_records\*.json | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | 
    Remove-Item
```

### 備份記錄
```powershell
# 壓縮所有記錄
Compress-Archive -Path data\training_records\* -DestinationPath backup_training_records.zip
```

---

## 🔍 常見問題

### Q: 舊記錄會丟失嗎？
A: 不會。您可以使用遷移工具轉換，或手動保留 `training_history.json`

### Q: 記錄會佔用很多空間嗎？
A: 不會。每個記錄文件約 0.1-0.5 KB，1000 條記錄約 100-500 KB

### Q: 可以自動清理舊記錄嗎？
A: 目前需手動清理，未來可能添加自動清理選項

### Q: 如何匯出記錄？
A: 直接複製 `data/training_records/` 文件夾即可

---

## 📝 版本歷史

### v2.0 (2026-02-08)
- ✅ 改用獨立文件存儲
- ✅ 時間戳文件命名
- ✅ 增加存儲統計
- ✅ 一鍵開啟資料夾
- ✅ 清除確認對話框

### v1.0 (舊版)
- 單一 JSON 文件存儲

---

## 🎉 總結

新的存儲系統更加靈活、可擴展，並且便於管理。如有任何問題，請查看 UI 中的提示或聯繫技術支援。
