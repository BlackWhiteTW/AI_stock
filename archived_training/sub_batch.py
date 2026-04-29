import streamlit as st
import pandas as pd
import time
from datetime import datetime
from utils import get_stock_data, get_tw_stock_list
from services.stock_strategy import StockStrategy
from pages_ui.components.shared import header_with_icon


def render_batch_training(ai_advisor):
    """批量股票模型訓練子模組"""
    header_with_icon("批量股票模型訓練", "📦", "自動執行多支股票的 AI 預測模型訓練")

    # 建立四種批量模式的列
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📦 批量熱門股")
        popular_stocks = ["0050.TW", "0051.TW", "2330.TW", "2317.TW", "2454.TW"]
        st.caption(f"訓練標的: {', '.join([s.split('.')[0] for s in popular_stocks])}")
        if st.button("🚀 開始批量訓練", key="batch_popular"):
            _run_simple_batch(ai_advisor, popular_stocks)

    with col2:
        st.markdown("### 🔥 智能篩選")
        st.caption("自動訓練技術指標評分高 (趨勢向上) 的強勢股")
        if st.button("🧠 啟動智能訓練", key="batch_smart"):
            _run_smart_batch(ai_advisor)

    with col3:
        st.markdown("### 🏆 全部優良股")
        st.caption("針對 AI 採購清單 (Shopping List) 中的標的進行訓練")
        if st.button("🏁 訓練優良股", key="batch_shopping"):
            _run_shopping_batch(ai_advisor)

    with col4:
        st.markdown("### 🌪️ 週補完計畫")
        st.caption("自動找出**一週內未訓練**的所有股票進行模型更新")
        if st.button("🔄 啟動全市場補完", key="batch_full_weekly"):
            _run_weekly_completion_batch(ai_advisor)


def _run_weekly_completion_batch(ai_advisor):
    """執行全市場週補完計畫"""
    import os
    from datetime import datetime, timedelta
    from utils import get_tw_stock_list, get_stock_data

    df_stocks = get_tw_stock_list()
    all_tickers = df_stocks["Ticker"].tolist()
    model_dir = "data/saved_models"

    to_train = []
    one_week_ago = datetime.now() - timedelta(days=7)

    # 1. 篩選需要訓練的名單
    for ticker in all_tickers:
        model_path = os.path.join(model_dir, f"{ticker}.pth")

        # 情況 A: 模型不存在
        if not os.path.exists(model_path):
            to_train.append(ticker)
            continue

        # 情況 B: 模型超過 7 天未更新
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        if mtime < one_week_ago:
            to_train.append(ticker)

    if not to_train:
        st.success("✅ 所有股票模型皆在 7 天內更新過，無需重複訓練！")
        return

    st.info(f"發現 {len(to_train)} 檔股票需要更新模型。啟動佇列訓練...")

    # 2. 啟動訓練
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, t in enumerate(to_train):
        status_text.text(f"⏳ 正在訓練 ({idx+1}/{len(to_train)}): {t}...")
        try:
            _, hist, _, _ = get_stock_data(
                t, force_offline=True
            )  # 優先使用本地數據加速
            if not hist.empty and len(hist) >= 60:
                ai_advisor.train_deep_learning_model(t, hist)
        except Exception as e:
            print(f"訓練 {t} 失敗: {e}")

        progress_bar.progress((idx + 1) / len(to_train))

    st.success(f"🎉 全市場補完計畫完成！共更新 {len(to_train)} 檔模型。")
    st.balloons()


def _run_simple_batch(ai_advisor, tickers):
    progress_bar = st.progress(0)
    for idx, t in enumerate(tickers):
        st.write(f"正在訓練: {t}...")
        _, hist, _, _ = get_stock_data(t)
        if not hist.empty:
            ai_advisor.train_deep_learning_model(t, hist)
        progress_bar.progress((idx + 1) / len(tickers))
    st.success("✅ 批量熱門股訓練完成！")


def _run_smart_batch(ai_advisor):
    df_stocks = get_tw_stock_list()
    tickers = df_stocks["Ticker"].tolist()
    analyzer = StockStrategy()

    success_count = 0
    progress_bar = st.progress(0)

    with st.status("正在進行全市場掃描與訓練...", expanded=True) as status:
        for idx, t in enumerate(tickers):
            _, hist, _, _ = get_stock_data(t, force_offline=True)
            if not hist.empty and len(hist) >= 60:
                result = analyzer.analyze_stock(t, hist)
                if result.get("score", 0) >= 5:
                    status.write(f"✨ {t} 評分達標 ({result['score']})，啟動訓練...")
                    ai_advisor.train_deep_learning_model(t, hist)
                    success_count += 1
            progress_bar.progress((idx + 1) / len(tickers))
    st.success(f"🎉 智能訓練完成！共訓練了 {success_count} 支強勢股。")


def _run_shopping_batch(ai_advisor):
    try:
        df = pd.read_csv("data/ai_shopping_list_basic.csv")
        tickers = [
            str(t).replace(".TW", "").replace(".TWO", "") for t in df["代碼"].tolist()
        ]
        _run_simple_batch(ai_advisor, tickers)
    except Exception as e:
        st.error(f"❌ 無法讀取清單: {e}")
