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
            st.metric("最後更新", mtime.strftime('%Y-%m-%d %H:%M'))
        else:
            st.warning("⚠️ 尚未訓練主模型")

def _run_base_training(service, enhanced):
    """執行訓練邏輯"""
    with st.spinner("正在進行基礎模型訓練..."):
        # 這裡簡化調用，實際會帶入進度回調
        result = service.train_base_model(enhanced=enhanced)
        st.success(result)
        st.balloons()
