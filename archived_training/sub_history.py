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

    # 讀取並清洗數據
    records = []
    for f in sorted(os.listdir(records_dir), reverse=True):
        if f.endswith(".json"):
            try:
                with open(os.path.join(records_dir, f), "r", encoding="utf-8") as file:
                    records.append(json.load(file))
            except:
                continue

    if not records:
        st.info("📝 尚無訓練記錄。")
        return

    df = pd.DataFrame(records)

    # 欄位映射與格式化
    column_mapping = {
        "timestamp": "訓練時間",
        "ticker": "股票代碼",
        "loss": "最終損失 (Loss)",
        "model_type": "模型類型",
        "version": "版本",
    }

    # 確保所有映射欄位都在 df 中，若無則補為 N/A
    for col in column_mapping.keys():
        if col not in df.columns:
            df[col] = "N/A"

    # 統計摘要區
    st.subheader("📋 紀錄概覽")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總訓練次數", len(df))
    col2.metric("涵蓋股票數", df["ticker"].nunique() if "ticker" in df.columns else 0)

    # 計算平均 Loss (排除 N/A)
    try:
        valid_loss = pd.to_numeric(df["loss"], errors="coerce").dropna()
        avg_loss = valid_loss.mean() if not valid_loss.empty else 0
        col3.metric("平均損失", f"{avg_loss:.6f}")
    except:
        col3.metric("平均損失", "N/A")

    latest_date = (
        df["timestamp"].max()[:10]
        if "timestamp" in df.columns and len(df["timestamp"]) > 0
        else "N/A"
    )
    col4.metric("最新訓練", latest_date)

    st.divider()

    # 篩選區
    st.subheader("🔍 篩選與過濾")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        # 修正：確保所有值都是字串以避免排序錯誤
        raw_types = (
            df["model_type"].dropna().unique().tolist()
            if "model_type" in df.columns
            else []
        )
        types = ["全部"] + sorted([str(t) for t in raw_types])
        sel_type = st.selectbox("模型類型", types)

    with f_col2:
        raw_tickers = (
            df["ticker"].dropna().unique().tolist() if "ticker" in df.columns else []
        )
        tickers = ["全部"] + sorted([str(t) for t in raw_tickers])
        sel_ticker = st.selectbox("標的搜尋", tickers)

    with f_col3:
        raw_versions = (
            df["version"].dropna().unique().tolist() if "version" in df.columns else []
        )
        versions = ["全部"] + sorted([str(v) for v in raw_versions])
        sel_version = st.selectbox("模型版本", versions)

    # 執行過濾
    df_filtered = df.copy()
    if sel_type != "全部":
        df_filtered = df_filtered[df_filtered["model_type"] == sel_type]
    if sel_ticker != "全部":
        df_filtered = df_filtered[df_filtered["ticker"] == sel_ticker]
    if sel_version != "全部":
        df_filtered = df_filtered[df_filtered["version"] == sel_version]

    # 重新整理顯示欄位與排序
    df_display = df_filtered.reindex(
        columns=["timestamp", "ticker", "model_type", "loss", "version"]
    )
    df_display = df_display.sort_values(by="timestamp", ascending=False)

    # 顯示表格
    st.dataframe(
        df_display,
        column_config={
            "timestamp": st.column_config.TextColumn("訓練時間"),
            "ticker": st.column_config.TextColumn("股票代碼"),
            "model_type": st.column_config.TextColumn("類型"),
            "loss": st.column_config.NumberColumn("Loss (越小越好)", format="%.6f"),
            "version": st.column_config.TextColumn("版本"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # 管理功能
    with st.expander("🛠️ 進階管理"):
        if st.button("🗑️ 永久清除所有歷史紀錄", type="secondary"):
            import shutil

            try:
                shutil.rmtree(records_dir)
                os.makedirs(records_dir)
                st.success("✅ 紀錄已永久清除")
                st.rerun()
            except Exception as e:
                st.error(f"清除失敗: {e}")
