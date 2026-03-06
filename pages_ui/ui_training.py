import streamlit as st
from .training.sub_base_model import render_base_model_training
from .training.sub_stock_model import render_stock_model_training
from .training.sub_gnn import render_gnn_training
from .training.sub_batch import render_batch_training
from .training.sub_history import render_training_history

def render_tab_training(ai_advisor):
    """
    🎓 AI 模型訓練中心 (彙整入口)
    負責將所有訓練相關的子模組組合成一個完整的分頁介面
    """
    st.header("🎓 AI 模型訓練中心")
    st.caption("管理和訓練 AI 深度學習模型，提升股票預測準確度")
    
    # 定義子分頁 (恢復完整功能)
    training_tab1, training_tab2, training_tab3, training_tab4, training_tab5 = st.tabs([
        "🚀 主模型訓練",
        "📈 個股模型訓練", 
        "🕸️ 市場關係訓練 (GNN)",
        "📦 批量訓練",
        "📊 訓練歷史與統計"
    ])
    
    with training_tab1:
        render_base_model_training()
    
    with training_tab2:
        render_stock_model_training(ai_advisor)

    with training_tab3:
        render_gnn_training()
        
    with training_tab4:
        render_batch_training(ai_advisor)
        
    with training_tab5:
        render_training_history()
