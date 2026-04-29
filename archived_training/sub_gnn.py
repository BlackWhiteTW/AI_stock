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
    """GNN 市場關係訓練介面 - 高度靈活版 v3.5"""
    header_with_icon("GNN 市場聯動關係訓練", "🕸️", "捕捉產業與隱藏關係的群體效應")

    gnn_service = GNNService()
    df_stocks = get_tw_stock_list()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### ⚙️ 訓練配置")

        # --- 模式切換 ---
        selection_mode = st.radio(
            "選取模式", ["📈 依市值排名 (Top N)", "🏭 依特定產業"], horizontal=True
        )

        tickers = []
        if selection_mode == "📈 依市值排名 (Top N)":
            max_val = len(df_stocks)
            num_stocks = st.slider("選取股票數量", 10, max_val, 100, step=10)
            tickers = df_stocks["Ticker"].head(num_stocks).tolist()
            st.caption(f"💡 將選取台股前 {num_stocks} 支大型權值股進行圖神經網路建構。")

        elif selection_mode == "🏭 依特定產業":
            all_industries = sorted(df_stocks["產業"].unique().tolist())
            selected_inds = st.multiselect(
                "選擇想要分析的產業群組",
                all_industries,
                default=["半導體業", "電腦及週邊設備業"],
            )
            tickers = df_stocks[df_stocks["產業"].isin(selected_inds)][
                "Ticker"
            ].tolist()
            st.caption(f"💡 目前選取了 {len(tickers)} 支屬於該族群的股票。")

        epochs = st.slider("訓練輪數 (Epochs)", 50, 1000, 100, step=50)

    with col2:
        # --- 資源與模型狀態診斷 ---
        st.markdown("### 📊 資源預估")
        if tickers:
            num_nodes = len(tickers)
            # 預估邊數 (假設平均每個節點連向同產業的其他節點，加上動態相關性)
            est_edges = num_nodes * 5

            st.info(
                f"""
            - **預計節點**: {num_nodes} 支股票
            - **預計邊數**: ~{est_edges} 條連動線
            - **運算負荷**: {"🟢 低" if num_nodes < 200 else "🟡 中" if num_nodes < 600 else "🔴 高"}
            """
            )

        if os.path.exists(gnn_service.model_path):
            st.success("✅ 現有模型已就緒")
            mtime = datetime.fromtimestamp(os.path.getmtime(gnn_service.model_path))
            st.caption(f"最後更新: {mtime.strftime('%Y-%m-%d %H:%M')}")

    if st.button(
        "🚀 啟動自定義 GNN 訓練",
        type="primary",
        use_container_width=True,
        disabled=not tickers,
    ):
        _execute_gnn_training(gnn_service, tickers, epochs)

    # 預覽功能
    if os.path.exists(gnn_service.model_path):
        _render_gnn_preview(gnn_service)


def _execute_gnn_training(gnn_service, tickers, epochs):
    """執行訓練邏輯"""
    status_placeholder = st.empty()
    with status_placeholder.status("🕸️ GNN 訓練初始化...", expanded=True) as status:
        try:

            def gnn_progress(msg, progress):
                status.write(msg)

            status.write(f"📋 已選取 {len(tickers)} 支股票進行訓練")
            result = gnn_service.train_market_gnn(
                tickers=tickers, epochs=epochs, progress_callback=gnn_progress
            )

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
                t_to_name = dict(zip(df_all["Ticker"], df_all["名稱"]))
                t_to_ind = dict(zip(df_all["Ticker"], df_all["產業"]))

                raw_scores = list(preds.values())
                max_s, min_s = max(raw_scores), min(raw_scores)
                range_s = max_s - min_s if max_s != min_s else 1.0

                display_data = []
                for ticker, score in preds.items():
                    strength = 20 + ((score - min_s) / range_s * 75)
                    display_data.append(
                        {
                            "代碼": ticker,
                            "股票名稱": t_to_name.get(ticker, "未知"),
                            "產業": t_to_ind.get(ticker, "其他"),
                            "看漲力道 (%)": round(strength, 2),
                        }
                    )

                df_preds = pd.DataFrame(display_data).sort_values(
                    by="看漲力道 (%)", ascending=False
                )

                # --- 頂部摘要 ---
                st.write("📈 **GNN 市場關係預測結果**")
                c1, c2, c3 = st.columns(3)
                c1.metric("最強勢標的", df_preds.iloc[0]["股票名稱"])
                c2.metric("平均看漲力道", f"{df_preds['看漲力道 (%)'].mean():.1f}%")

                # --- 圖表 ---
                df_ind = (
                    df_preds.groupby("產業")["看漲力道 (%)"]
                    .mean()
                    .reset_index()
                    .sort_values(by="看漲力道 (%)")
                )
                fig = px.bar(
                    df_ind,
                    x="看漲力道 (%)",
                    y="產業",
                    orientation="h",
                    color="看漲力道 (%)",
                    color_continuous_scale="RdYlGn",
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- 表格 ---
                st.dataframe(df_preds, use_container_width=True, hide_index=True)
