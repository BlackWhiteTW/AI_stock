"""
📊 額度管理服務 (Quota Manager) v1.0
負責追蹤 API 使用次數，保護使用者的錢包與額度
"""
import os
import json
from datetime import datetime
from pathlib import Path

class QuotaManager:
    def __init__(self, limit=250):
        self.limit = limit
        self.file_path = Path("data/api_quota.json")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()

    def _init_file(self):
        if not self.file_path.exists():
            self._save_data({"count": 0, "last_reset": datetime.now().strftime("%Y-%m")})

    def _save_data(self, data):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _get_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 每月自動重置
            current_month = datetime.now().strftime("%Y-%m")
            if data.get("last_reset") != current_month:
                data = {"count": 0, "last_reset": current_month}
                self._save_data(data)
            return data
        except:
            return {"count": 0, "last_reset": datetime.now().strftime("%Y-%m")}

    def log_request(self):
        """記錄一次請求"""
        data = self._get_data()
        data["count"] += 1
        self._save_data(data)

    def get_status(self):
        """獲取當前狀態"""
        data = self._get_data()
        usage = data["count"]
        remaining = max(0, self.limit - usage)
        pct = (usage / self.limit) * 100
        return {
            "usage": usage,
            "limit": self.limit,
            "remaining": remaining,
            "percent": pct,
            "is_low": pct > 80
        }
