import streamlit as st
import os
import time
from datetime import datetime
from services.ai_service import AIService
from pages_ui.components.shared import header_with_icon, status_card


def render_base_model_training():
    """主模型訓練子模組"""
    header_with_icon("遷移學習主模型訓練", "🚀", "提升個股訓練速度與準確度的基礎模型")

    service = AIService()
    model_path = "data/saved_models/base_model.pth"

    col1, col2 = st.columns(2)
    with col1:
        st.info("⏱️ 主模型訓練包含大量權值股數據，通常需 15-30 分鐘。")
        mode = st.radio("模式", ["🚀 增強版", "⚡ 標準版"], key="base_mode")
        if st.button("🎓 啟動主模型訓練", type="primary"):
            _run_base_training(service, mode == "🚀 增強版")

    with col2:
        if os.path.exists(model_path):
            st.success("✅ 主模型已就緒")
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            st.metric("最後更新", mtime.strftime("%Y-%m-%d %H:%M"))
        else:
            st.warning("⚠️ 尚未訓練主模型")


def _run_base_training(service, enhanced):
    """執行訓練邏輯"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()

    def ui_progress_callback(current, total, loss, time_elapsed, ticker=None):
        if ticker:  # 數據準備階段
            if current == 0 and total == 100:
                status_text.text(f"📦 {ticker}...")
                progress_bar.progress(0.05)
            else:
                progress = (current / total) * 0.3  # 數據準備佔 30%
                status_text.text(f"📥 數據準備中: {ticker} ({current}/{total})")
                progress_bar.progress(progress)
        else:  # 模型訓練階段
            progress = 0.3 + (current / total) * 0.7  # 訓練佔 70%
            progress_bar.progress(progress)
            status_text.text(f"🎓 模型訓練中: 第 {current}/{total} 輪")
            with metrics_placeholder.container():
                c1, c2 = st.columns(2)
                c1.metric("當前 Loss", f"{loss:.6f}")
                c2.metric("訓練進度", f"{int((current/total)*100)}%")

    # 執行訓練
    result = service.train_base_model(
        enhanced=enhanced, progress_callback=ui_progress_callback
    )

    # 清理並顯示結果
    progress_bar.progress(1.0)
    status_text.success(result)
    metrics_placeholder.empty()
    st.balloons()
