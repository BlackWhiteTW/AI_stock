import streamlit as st
import os
import pandas as pd
import time
import plotly.express as px
from datetime import datetime
from services.gnn_service import GNNService
from utils import get_tw_stock_list
from pages_ui.components.shared import header_with_icon, status_card, info_box

def render_gnn_training():
    """GNN 市場關係訓練介面 - 子模組化 v3.0"""
    header_with_icon("GNN 市場聯動關係訓練", "🕸️", "捕捉產業與隱藏關係的群體效應")
    
    gnn_service = GNNService()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 GNN 會自動尋找相關性高的標的，並建立融合靜態產業與動態網路情緒的圖譜。")
        scope = st.selectbox("訓練範圍", ["精選前 100 支", "熱門前 500 支", "全市場級別"], key="gnn_scope")
        epochs = st.slider("訓練輪數", 50, 500, 100, key="gnn_epochs")
    
    with col2:
        if os.path.exists(gnn_service.model_path):
            stat = os.stat(gnn_service.model_path)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            st.success("✅ GNN 模型已就緒")
            st.metric("最後更新", mtime.strftime('%Y-%m-%d %H:%M'))
            st.metric("模型大小", f"{stat.st_size / 1024:.1f} KB")
        else:
            st.warning("⚠️ 尚未訓練市場 GNN 模型")

    if st.button("🚀 啟動 GNN 訓練演算法", type="primary", key="gnn_train_btn"):
        _execute_gnn_training(gnn_service, scope, epochs)

    # 預覽功能
    if os.path.exists(gnn_service.model_path):
        _render_gnn_preview(gnn_service)

def _execute_gnn_training(gnn_service, scope, epochs):
    """執行訓練邏輯"""
    status_placeholder = st.empty()
    with status_placeholder.status("🕸️ GNN 訓練初始化...", expanded=True) as status:
        try:
            def gnn_progress(msg, progress):
                status.write(msg)
            
            df_stocks = get_tw_stock_list()
            if scope == "精選前 100 支": tickers = df_stocks['Ticker'].head(100).tolist()
            elif scope == "熱門前 500 支": tickers = df_stocks['Ticker'].head(500).tolist()
            else: tickers = df_stocks['Ticker'].tolist()
            
            status.write(f"📋 已選取 {len(tickers)} 支股票進行訓練")
            result = gnn_service.train_market_gnn(tickers=tickers, epochs=epochs, progress_callback=gnn_progress)
            
            status.update(label="✅ GNN 訓練完成！", state="complete", expanded=False)
            st.success(result)
            st.balloons()
            st.rerun()
        except Exception as e:
            status.update(label="❌ GNN 訓練失敗", state="error")
            st.error(f"GNN 訓練失敗: {e}")

def _render_gnn_preview(gnn_service):
    """渲染預覽圖表與報告"""
    if st.button("🔍 預覽 GNN 市場關係報告", key="gnn_preview_btn"):
        with st.spinner("正在提取圖特徵進行推論..."):
            preds = gnn_service.predict_market_relations()
            
            if isinstance(preds, dict) and "error" in preds:
                st.error(preds["message"])
            elif isinstance(preds, dict) and len(preds) > 0:
                # --- 數據處理邏輯 ---
                df_all = get_tw_stock_list()
                t_to_name = dict(zip(df_all['Ticker'], df_all['名稱']))
                t_to_ind = dict(zip(df_all['Ticker'], df_all['產業']))
                
                raw_scores = list(preds.values())
                max_s, min_s = max(raw_scores), min(raw_scores)
                range_s = max_s - min_s if max_s != min_s else 1.0

                display_data = []
                for ticker, score in preds.items():
                    strength = 20 + ((score - min_s) / range_s * 75)
                    display_data.append({
                        "代碼": ticker,
                        "股票名稱": t_to_name.get(ticker, "未知"),
                        "產業": t_to_ind.get(ticker, "其他"),
                        "看漲力道 (%)": round(strength, 2)
                    })
                
                df_preds = pd.DataFrame(display_data).sort_values(by="看漲力道 (%)", ascending=False)
                
                # --- 頂部摘要 ---
                st.write("📈 **GNN 市場關係預測結果**")
                c1, c2, c3 = st.columns(3)
                c1.metric("最強勢標的", df_preds.iloc[0]['股票名稱'])
                c2.metric("平均看漲力道", f"{df_preds['看漲力道 (%)'].mean():.1f}%")
                
                # --- 圖表 ---
                df_ind = df_preds.groupby('產業')['看漲力道 (%)'].mean().reset_index().sort_values(by='看漲力道 (%)')
                fig = px.bar(df_ind, x='看漲力道 (%)', y='產業', orientation='h', color='看漲力道 (%)', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 表格 ---
                st.dataframe(df_preds, use_container_width=True, hide_index=True)
