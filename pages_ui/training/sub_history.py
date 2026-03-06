import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from pages_ui.components.shared import header_with_icon

def render_training_history():
    """訓練歷史與統計子模組"""
    header_with_icon("訓練歷史與模型統計", "📊", "追蹤 AI 模型的訓練成效與存儲狀態")
    
    records_dir = "data/training_records"
    if not os.path.exists(records_dir):
        st.info("📝 尚無訓練記錄。")
        return

    # 讀取所有記錄
    records = []
    for f in sorted(os.listdir(records_dir), reverse=True):
        if f.endswith('.json'):
            with open(os.path.join(records_dir, f), 'r', encoding='utf-8') as file:
                records.append(json.load(file))

    if not records:
        st.info("📝 尚無訓練記錄。")
        return

    # 統計摘要
    st.subheader("📋 紀錄概覽")
    df = pd.DataFrame(records)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("總訓練次數", len(df))
    col2.metric("涵蓋股票數", df['ticker'].nunique() if 'ticker' in df.columns else "N/A")
    col3.metric("最新訓練日期", df['timestamp'].iloc[0][:10] if 'timestamp' in df.columns else "N/A")

    st.divider()
    
    # 顯示表格
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 管理功能
    if st.button("🗑️ 清除歷史紀錄"):
        import shutil
        shutil.rmtree(records_dir)
        os.makedirs(records_dir)
        st.success("✅ 紀錄已清除")
        st.rerun()
