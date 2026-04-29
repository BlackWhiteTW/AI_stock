"""
⚡ 統一高效能快取管理器 (Unified Cache Manager) v1.0
解決檔案過多導致的效能問題
核心技術：SQLite + Parquet + LRU Memory Cache
"""

import os
import json
import sqlite3
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import OrderedDict
import pandas as pd
import threading


class CacheManager:
    """
    統一快取管理器

    特點：
    1. 三層快取架構：Memory -> SQLite -> Parquet
    2. 自動索引和查詢優化
    3. LRU 淘汰策略
    4. 線程安全
    5. 自動清理過期資料
    """

    def __init__(self, cache_dir="data/unified_cache", max_memory_items=1000):
        """
        初始化快取管理器

        Args:
            cache_dir: 快取根目錄
            max_memory_items: 內存快取最大項目數
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite 資料庫（用於小型數據和索引）
        self.db_path = self.cache_dir / "cache_index.db"
        self.conn = None
        self.lock = threading.Lock()

        # Parquet 目錄（用於大型 DataFrame）
        self.parquet_dir = self.cache_dir / "parquet"
        self.parquet_dir.mkdir(exist_ok=True)

        # 內存快取（LRU）
        self.memory_cache = OrderedDict()
        self.max_memory_items = max_memory_items

        # 初始化資料庫
        self._init_database()

    def _init_database(self):
        """初始化 SQLite 資料庫結構"""
        with self.lock:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.conn.cursor()

            # 主快取表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    category TEXT,
                    data_type TEXT,
                    value BLOB,
                    parquet_path TEXT,
                    created_at REAL,
                    updated_at REAL,
                    expires_at REAL,
                    access_count INTEGER DEFAULT 0,
                    last_access REAL
                )
            """
            )

            # 創建索引以加速查詢
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_category 
                ON cache_entries(category)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires 
                ON cache_entries(expires_at)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_access 
                ON cache_entries(last_access)
            """
            )

            self.conn.commit()

    def _generate_key(self, category: str, identifier: str) -> str:
        """生成快取鍵值"""
        return f"{category}:{identifier}"

    def _get_parquet_path(self, key: str) -> Path:
        """獲取 Parquet 檔案路徑"""
        # 使用 hash 分散檔案到子目錄
        hash_prefix = hashlib.md5(key.encode()).hexdigest()[:2]
        subdir = self.parquet_dir / hash_prefix
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key.replace(':', '_')}.parquet"

    def set(
        self,
        category: str,
        identifier: str,
        value: Any,
        ttl: int = 3600,
        data_type: str = "auto",
    ) -> bool:
        """
        設置快取

        Args:
            category: 類別（如 'stock_data', 'news', 'prediction'）
            identifier: 識別符（如 '2330.TW', 'weekly_news'）
            value: 要快取的值
            ttl: 存活時間（秒），0 表示永不過期
            data_type: 數據類型 ('json', 'dataframe', 'pickle', 'auto')

        Returns:
            是否成功
        """
        key = self._generate_key(category, identifier)

        # 自動判斷數據類型
        if data_type == "auto":
            if isinstance(value, pd.DataFrame):
                data_type = "dataframe"
            elif isinstance(value, (dict, list)):
                data_type = "json"
            else:
                data_type = "pickle"

        now = datetime.now().timestamp()
        expires_at = now + ttl if ttl > 0 else 0

        with self.lock:
            try:
                # DataFrame 存儲到 Parquet
                if data_type == "dataframe":
                    parquet_path = self._get_parquet_path(key)
                    value.to_parquet(parquet_path, compression="snappy")

                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries 
                        (key, category, data_type, parquet_path, created_at, updated_at, expires_at, last_access)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            category,
                            data_type,
                            str(parquet_path),
                            now,
                            now,
                            expires_at,
                            now,
                        ),
                    )

                else:
                    # 小型數據存儲到 SQLite
                    if data_type == "json":
                        blob = json.dumps(value).encode("utf-8")
                    else:
                        blob = pickle.dumps(value)

                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries 
                        (key, category, data_type, value, created_at, updated_at, expires_at, last_access)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (key, category, data_type, blob, now, now, expires_at, now),
                    )

                self.conn.commit()

                # 更新內存快取
                self._update_memory_cache(key, value)

                return True

            except Exception as e:
                print(f"❌ 快取設置失敗 {key}: {e}")
                return False

    def get(self, category: str, identifier: str, default: Any = None) -> Any:
        """
        獲取快取

        Args:
            category: 類別
            identifier: 識別符
            default: 預設值（找不到時返回）

        Returns:
            快取的值或預設值
        """
        key = self._generate_key(category, identifier)

        # 1. 先檢查內存快取
        if key in self.memory_cache:
            self._update_memory_cache(key, self.memory_cache[key])
            return self.memory_cache[key]

        # 2. 從 SQLite 讀取
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    SELECT data_type, value, parquet_path, expires_at 
                    FROM cache_entries 
                    WHERE key = ?
                """,
                    (key,),
                )

                row = cursor.fetchone()

                if not row:
                    return default

                data_type, value_blob, parquet_path, expires_at = row

                # 檢查是否過期
                if expires_at > 0 and datetime.now().timestamp() > expires_at:
                    self.delete(category, identifier)
                    return default

                # 讀取數據
                if data_type == "dataframe":
                    value = pd.read_parquet(parquet_path)
                elif data_type == "json":
                    value = json.loads(value_blob.decode("utf-8"))
                else:
                    value = pickle.loads(value_blob)

                # 更新訪問統計
                cursor.execute(
                    """
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_access = ?
                    WHERE key = ?
                """,
                    (datetime.now().timestamp(), key),
                )
                self.conn.commit()

                # 更新內存快取
                self._update_memory_cache(key, value)

                return value

            except Exception as e:
                print(f"❌ 快取讀取失敗 {key}: {e}")
                return default

    def _update_memory_cache(self, key: str, value: Any):
        """更新內存快取（LRU 策略）"""
        # 如果已存在，先移除（移到最後）
        if key in self.memory_cache:
            del self.memory_cache[key]

        # 添加到最後
        self.memory_cache[key] = value

        # 檢查是否超過大小限制
        if len(self.memory_cache) > self.max_memory_items:
            # 移除最舊的項目
            self.memory_cache.popitem(last=False)

    def delete(self, category: str, identifier: str) -> bool:
        """刪除快取"""
        key = self._generate_key(category, identifier)

        with self.lock:
            try:
                cursor = self.conn.cursor()

                # 獲取 Parquet 路徑（如果有）
                cursor.execute(
                    "SELECT parquet_path FROM cache_entries WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row and row[0]:
                    parquet_path = Path(row[0])
                    if parquet_path.exists():
                        parquet_path.unlink()

                # 從資料庫刪除
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self.conn.commit()

                # 從內存快取刪除
                if key in self.memory_cache:
                    del self.memory_cache[key]

                return True

            except Exception as e:
                print(f"❌ 快取刪除失敗 {key}: {e}")
                return False

    def clear_category(self, category: str) -> int:
        """清除某個類別的所有快取"""
        with self.lock:
            try:
                cursor = self.conn.cursor()

                # 獲取所有該類別的 Parquet 檔案
                cursor.execute(
                    "SELECT parquet_path FROM cache_entries WHERE category = ?",
                    (category,),
                )
                rows = cursor.fetchall()

                # 刪除 Parquet 檔案
                for row in rows:
                    if row[0]:
                        parquet_path = Path(row[0])
                        if parquet_path.exists():
                            parquet_path.unlink()

                # 從資料庫刪除
                cursor.execute(
                    "DELETE FROM cache_entries WHERE category = ?", (category,)
                )
                count = cursor.rowcount
                self.conn.commit()

                # 從內存快取刪除
                keys_to_remove = [
                    k for k in self.memory_cache if k.startswith(f"{category}:")
                ]
                for k in keys_to_remove:
                    del self.memory_cache[k]

                return count

            except Exception as e:
                print(f"❌ 類別清除失敗 {category}: {e}")
                return 0

    def cleanup_expired(self) -> int:
        """清理過期快取"""
        with self.lock:
            try:
                now = datetime.now().timestamp()
                cursor = self.conn.cursor()

                # 獲取過期項目
                cursor.execute(
                    """
                    SELECT key, parquet_path 
                    FROM cache_entries 
                    WHERE expires_at > 0 AND expires_at < ?
                """,
                    (now,),
                )
                rows = cursor.fetchall()

                # 刪除 Parquet 檔案
                for key, parquet_path in rows:
                    if parquet_path:
                        path = Path(parquet_path)
                        if path.exists():
                            path.unlink()

                    # 從內存快取刪除
                    if key in self.memory_cache:
                        del self.memory_cache[key]

                # 從資料庫刪除
                cursor.execute(
                    "DELETE FROM cache_entries WHERE expires_at > 0 AND expires_at < ?",
                    (now,),
                )
                count = cursor.rowcount
                self.conn.commit()

                return count

            except Exception as e:
                print(f"❌ 過期清理失敗: {e}")
                return 0

    def get_stats(self) -> Dict[str, Any]:
        """獲取快取統計資訊"""
        with self.lock:
            try:
                cursor = self.conn.cursor()

                # 總項目數
                cursor.execute("SELECT COUNT(*) FROM cache_entries")
                total_count = cursor.fetchone()[0]

                # 各類別統計
                cursor.execute(
                    """
                    SELECT category, COUNT(*), data_type
                    FROM cache_entries 
                    GROUP BY category, data_type
                """
                )
                category_stats = {}
                for category, count, data_type in cursor.fetchall():
                    if category not in category_stats:
                        category_stats[category] = {}
                    category_stats[category][data_type] = count

                # 內存快取使用
                memory_usage = len(self.memory_cache)

                # 磁碟使用
                total_size = 0
                for file in self.cache_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

                return {
                    "total_entries": total_count,
                    "memory_cache_size": memory_usage,
                    "disk_size_mb": total_size / (1024 * 1024),
                    "categories": category_stats,
                }

            except Exception as e:
                print(f"❌ 統計獲取失敗: {e}")
                return {}

    def batch_get(self, category: str, identifiers: List[str]) -> Dict[str, Any]:
        """批量獲取快取（效能優化）"""
        results = {}
        for identifier in identifiers:
            value = self.get(category, identifier)
            if value is not None:
                results[identifier] = value
        return results

    def batch_set(self, category: str, items: Dict[str, Any], ttl: int = 3600):
        """批量設置快取（效能優化）"""
        for identifier, value in items.items():
            self.set(category, identifier, value, ttl=ttl)

    def close(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()


# 全局單例
_cache_manager_instance = None


def get_cache_manager() -> CacheManager:
    """獲取全局快取管理器單例"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance
