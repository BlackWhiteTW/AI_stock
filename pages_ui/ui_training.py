import streamlit as st
import pandas as pd
import time
import os
import json
import subprocess
from datetime import datetime
from services.ai_service import AIService
from services.gnn_service import GNNService
from services.stock_strategy import StockStrategy
from utils import get_stock_data, get_tw_stock_list

def render_tab_training(ai_advisor):
    # ... (其餘 UI 邏輯保持不變)
    """渲染 AI 模型訓練中心頁面"""
    
    st.header("🎓 AI 模型訓練中心")
    st.caption("管理和訓練 AI 深度學習模型，提升股票預測準確度")
    
    # 創建子分頁（5個）
    training_tab1, training_tab2, training_tab3, training_tab4, training_tab5 = st.tabs([
        "🚀 主模型訓練",
        "📈 個股模型訓練 (LSTM)", 
        "🕸️ 市場關係訓練 (GNN)",
        "📦 批量訓練",
        "📊 訓練歷史與統計"
    ])
    
    # ===== Tab 1: 主模型訓練 =====
    with training_tab1:
        render_base_model_training()
    
    # ===== Tab 2: 個股模型訓練 =====
    with training_tab2:
        render_stock_model_training(ai_advisor)

    # ===== Tab 3: GNN 市場關係訓練 =====
    with training_tab3:
        render_gnn_training()
    
    # ===== Tab 4: 批量訓練 =====
    with training_tab4:
        render_batch_training(ai_advisor)
    
    # ===== Tab 5: 訓練歷史 =====
    with training_tab5:
        render_training_history()

def render_gnn_training():
    """GNN 市場關係訓練介面"""
    st.subheader("🕸️ GNN 市場聯動關係訓練")
    st.markdown("""
    **圖神經網路 (GNN)** 用於捕捉股票之間的「群體效應」：
    - 💎 **產業聯動**：同一產業的股票通常會一起漲跌。
    - 🔗 **關係權重**：自動學習哪些股票是市場的領頭羊。
    - 📊 **全局預測**：不再孤立看單一股票，而是看整個市場圖譜。
    """)
    
    from services.gnn_service import GNNService
    gnn_service = GNNService()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 GNN 會根據『產業分類』將股票連接成網路，並學習節點間的特徵傳遞。")
        scope = st.selectbox("訓練範圍 (Market Scope)", ["精選前 100 支", "熱門前 500 支", "全市場級別 (所有上市櫃)"])
        epochs = st.slider("訓練輪數 (Epochs)", 50, 500, 100)
    
    with col2:
        # 檢查 GNN 模型狀態
        if os.path.exists(gnn_service.model_path):
            stat = os.stat(gnn_service.model_path)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            st.success(f"✅ GNN 模型已就緒")
            st.metric("最後更新", mtime.strftime('%Y-%m-%d %H:%M'))
            st.metric("模型大小", f"{stat.st_size / 1024:.1f} KB")
        else:
            st.warning("⚠️ 尚未訓練市場 GNN 模型")
            st.info("建議先進行全市場關係訓練，以輔助 LSTM 選股。")

    if st.button("🚀 啟動市場關係訓練 (GNN)", type="primary"):
        with st.spinner(f"正在分析 {scope} 數據並構建圖結構..."):
            try:
                df_stocks = get_tw_stock_list()
                if scope == "精選前 100 支":
                    tickers = df_stocks['Ticker'].head(100).tolist()
                elif scope == "熱門前 500 支":
                    tickers = df_stocks['Ticker'].head(500).tolist()
                else:
                    tickers = df_stocks['Ticker'].tolist()
                
                result = gnn_service.train_market_gnn(tickers=tickers, epochs=epochs)
                st.success(result)
                st.balloons()
            except Exception as e:
                st.error(f"GNN 訓練失敗: {e}")

    # 預覽功能
    if os.path.exists(gnn_service.model_path):
        if st.button("🔍 預覽 GNN 市場關係預測"):
            with st.spinner("正在提取圖特徵進行推論..."):
                preds = gnn_service.predict_market_relations()
                if isinstance(preds, dict):
                    st.write("📈 **GNN 預測明日最強勢標的 (前 10 名)：**")
                    # 排序並顯示
                    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
                    df_preds = pd.DataFrame(sorted_preds[:10], columns=['股票', '預計強度 (GNN)'])
                    st.table(df_preds)


def render_base_model_training():
    """主模型訓練介面"""
    
    st.subheader("🚀 遷移學習主模型訓練")
    st.markdown("""
    訓練基礎模型可以：
    - ✅ 使用 25+ 支大型權值股和國際指數的數據
    - ✅ 提升所有個股的訓練速度 50-70%
    - ✅ 增強模型泛化能力和預測準確度
    - ✅ 涵蓋台股、ETF、國際市場多維度數據
    """)
    
    st.info("⏱️ **訓練時間說明**: 主模型訓練數據量約為個股的 100-200 倍，因此訓練時間通常為 **5-30 分鐘**（個股約 1-5 分鐘）。但訓練一次後，所有個股都能享受加速效果！")
    
    st.divider()
    
    # 訓練模式選擇
    col_mode1, col_mode2 = st.columns([1, 2])
    
    with col_mode1:
        training_mode = st.radio(
            "訓練模式",
            ["🚀 增強版（推薦）", "⚡ 標準版"],
            help="增強版使用更多股票（25支）、更長歷史（5000天）、國際指數，訓練時間較長但效果更好"
        )
    
    with col_mode2:
        if training_mode == "🚀 增強版（推薦）":
            st.info(
                "**增強版特色：**\n"
                "📊 25支標的（台股權值股 + ETF + 國際指數）\n"
                "📈 5000天歷史數據（約13年）\n"
                "🔧 數據增強技術（噪聲注入）\n"
                "🌍 國際市場視野（標普500、那斯達克）\n"
                "⏱️ 預計訓練時間：15-30分鐘"
            )
        else:
            st.info(
                "**標準版特色：**\n"
                "📊 8支台股大型權值股\n"
                "📈 2000天歷史數據（約5年）\n"
                "⏱️ 預計訓練時間：5-15分鐘"
            )
    
    st.divider()
    
    # 檢查現有模型
    model_status = check_base_model_status()
    
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        if model_status['exists']:
            st.success("✅ 主模型已訓練")
        else:
            st.warning("⚠️ 尚未訓練主模型")
    
    with col_status2:
        if model_status['last_trained']:
            st.metric("最後訓練時間", model_status['last_trained'])
        else:
            st.metric("最後訓練時間", "從未訓練")
    
    with col_status3:
        if model_status['file_size']:
            st.metric("模型大小", model_status['file_size'])
        else:
            st.metric("模型大小", "-")
    
    st.divider()
    
    # 訓練按鈕
    col_train1, col_train2, col_train3 = st.columns([1, 2, 1])
    
    with col_train1:
        enhanced = (training_mode == "🚀 增強版（推薦）")
        button_text = "🎓 開始訓練" if not enhanced else "🚀 開始增強訓練"
        
        if st.button(button_text, type="primary", use_container_width=True):
            st.session_state['start_base_training'] = True
            st.session_state['training_mode'] = enhanced
    
    # 執行訓練
    if st.session_state.get('start_base_training', False):
        enhanced = st.session_state.get('training_mode', False)
        
        with col_train2:
            if enhanced:
                st.warning("⏳ 增強版訓練中，預計需要 15-30 分鐘，請耐心等待...")
            else:
                st.info("⏳ 標準版訓練中，預計需要 5-15 分鐘...")
        
        # 訓練進度容器
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_text = st.empty()
            detail_expander = st.expander("📋 訓練詳細日誌", expanded=False)
            
            with detail_expander:
                log_text = st.empty()
            
            # 初始化 AI Service
            service = AIService()
            
            # 收集訓練日誌
            training_logs = []
            
            # 訓練進度回調
            def progress_callback(current, total, loss, elapsed, ticker=None):
                # 數據收集階段 (Loss 為 0 且在第一階段)
                if loss <= 0:
                    progress = (current / total) * 0.3
                    progress_bar.progress(min(progress, 0.3))
                    
                    ticker_display = f" ({ticker})" if ticker else ""
                    status_text.text(f"📥 數據收集中: {current}/{total} 支標的{ticker_display}...")

                    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載第 {current}/{total} 支標的數據{ticker_display}..."
                    if not training_logs or log_msg != training_logs[-1]:
                        training_logs.append(log_msg)
                        log_text.text("\n".join(training_logs[-20:]))
                else:
                    # 訓練階段（30%-100%）
                    progress = 0.3 + (0.7 * (current / total))
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"🧠 神經網絡訓練中 (第 {current}/{total} 輪)...")
                    stats_text.text(f"訓練進度 | Loss: {loss:.6f} | 預計還需: {int((total-current)*1.2)}s")

                    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {current}/{total} - Loss: {loss:.6f}"
                    if not training_logs or log_msg != training_logs[-1]:
                        training_logs.append(log_msg)
                        log_text.text("\n".join(training_logs[-20:]))
            try:
                # 開始訓練
                mode_text = "增強版" if enhanced else "標準版"
                start_time = time.time()
                
                with st.spinner(f"🎯 正在執行{mode_text}訓練..."):
                    result = service.train_base_model(
                        stock_list=None,
                        progress_callback=progress_callback,
                        enhanced=enhanced
                    )
                
                end_time = time.time()
                training_duration = end_time - start_time
                
                progress_bar.progress(1.0)
                status_text.text("✅ 訓練完成！")
                stats_text.text(f"總耗時: {training_duration/60:.1f} 分鐘")
                
                # 顯示結果
                st.success(result)
                st.balloons()
                
                # 保存訓練記錄
                save_training_record({
                    'type': 'base_model',
                    'mode': mode_text,
                    'enhanced': enhanced,
                    'duration': training_duration,
                    'timestamp': datetime.now().isoformat(),
                    'result': result
                })
                
                # 顯示詳細統計
                with st.expander("📊 訓練統計詳情", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("訓練模式", mode_text)
                    with col2:
                        if enhanced:
                            st.metric("標的數量", "200 支")
                        else:
                            st.metric("標的數量", "30 支")
                    with col3:
                        if enhanced:
                            st.metric("歷史數據", "5000+ 天")
                        else:
                            st.metric("歷史數據", "~2000 天")
                    with col4:
                        st.metric("訓練耗時", f"{training_duration/60:.1f} 分")
                
            except Exception as e:
                st.error(f"❌ 訓練失敗: {str(e)}")
                import traceback
                with st.expander("錯誤詳情"):
                    st.code(traceback.format_exc())
            
            finally:
                # 訓練完成後重置狀態
                st.session_state['start_base_training'] = False
                if 'training_mode' in st.session_state:
                    del st.session_state['training_mode']


def render_stock_model_training(ai_advisor):
    """個股模型訓練介面（增強版：加入即時分析預覽）"""
    
    st.subheader("📈 個股深度學習模型訓練")
    st.markdown("""
    針對特定股票訓練專屬的 GRU/LSTM 深度神經網路模型
    """)
    
    # 檢查是否有主模型
    base_model_exists = os.path.exists("data/saved_models/base_model.pth")
    if base_model_exists:
        st.success("🚀 **遷移學習已啟用** - 訓練速度提升 50-70%")
    else:
        st.warning("⚠️ **未找到主模型** - 建議先訓練主模型以加速訓練")
    
    st.divider()
    
    # 股票輸入區塊
    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        ticker_input = st.text_input(
            "輸入股票代碼",
            placeholder="例如: 2330, 2454, 0050",
            help="輸入台股代碼（會自動補上 .TW）"
        )
    
    # 用於存儲分析結果的變數
    analysis_done = False
    current_score = 0
    ticker = ""

    # 預處理股票代碼與執行戰略分析
    if ticker_input:
        # 只保留數字代碼，不自動補 .TW
        ticker = ticker_input.replace('.TW', '').replace('.TWO', '')
        
        # --- [新增] 自動執行策略分析 ---
        try:
            from utils import get_stock_data
            from services.stock_strategy import StockStrategy
            
            # 獲取少量數據進行快速分析
            _, hist_check, current_price, info = get_stock_data(ticker, period="1y") 
            
            if not hist_check.empty and len(hist_check) >= 60:
                # [修正] 不傳參數，避免 __init__ 報錯
                analyzer = StockStrategy()
                result = analyzer.analyze_stock(ticker, hist_check)
                
                score = result.get('score', 0)
                action = result.get('action', '未知')
                reasons = result.get('reasons', [])
                
                # 顯示分析摘要卡片
                with st.container():
                    stock_name = info.get('longName', '') or info.get('shortName', ticker)
                    st.markdown(f"**📊 {stock_name} ({ticker}) 技術面快篩**")
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        st.metric("當前價格", f"{current_price:.2f}")
                    with c2:
                        # 根據分數變色
                        color = "red"
                        if score >= 5: color = "green"
                        elif score >= 0: color = "orange"
                        st.markdown(f"評分：<span style='color:{color};font-size:20px'><b>{score} 分</b></span> ({action})", unsafe_allow_html=True)
                    with c3:
                        if score >= 5:
                            st.markdown("✅ **推薦訓練**")
                        elif score <= -2:
                            st.markdown("❌ **結構轉弱**")
                        else:
                            st.markdown("⚠️ **需觀察**")
                    
                    # 顯示主要理由 (只顯示前兩條)
                    if reasons:
                        with st.expander("查看詳細分析理由", expanded=False):
                            for r in reasons:
                                st.text(r)
                
                analysis_done = True
                current_score = score
            else:
                st.warning("⚠️ 數據不足，無法進行技術面評分，但仍可強制訓練。")
                
        except Exception as e:
            # 不阻擋訓練，只顯示警告
            pass

    with col_input2:
        st.write("") 
        st.write("")
        # 按鈕：如果分析過且分數低，顯示警告提示
        btn_label = "🧠 啟動訓練"
        if analysis_done and current_score < 0:
            btn_label = "⚠️ 強制啟動訓練"
            
        train_button = st.button(btn_label, type="primary", use_container_width=True, disabled=not ticker_input)
    
    # 執行訓練邏輯
    if train_button and ticker:
        st.divider()
        st.info(f"🎯 正在訓練 {ticker} 的 AI 預測模型...")
        
        # 訓練進度容器
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.empty()
            
            try:
                from utils import get_stock_data
                
                # 獲取股票數據
                status_text.text("📥 下載股票歷史數據...")
                _, hist, current_price, info = get_stock_data(ticker)
                
                if hist.empty:
                    st.error(f"❌ 無法獲取 {ticker} 的歷史數據，請檢查代碼是否正確")
                else:
                    progress_bar.progress(0.1)
                    stock_name = info.get('longName', '') or info.get('shortName', ticker)
                    
                    # 顯示股票基本信息
                    with st.expander("📊 股票基本信息", expanded=True):
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.metric("股票名稱", stock_name)
                        with info_col2:
                            st.metric("當前價格", f"${current_price:.2f}")
                        with info_col3:
                            st.metric("歷史數據", f"{len(hist)} 天")
                    
                    # 訓練配置
                    metrics = {"best_loss": float('inf')}
                    training_logs = []
                    
                    # 訓練進度回調
                    def on_train_progress(epoch, total_epochs, loss, train_elapsed=None):
                        progress = float(epoch) / float(total_epochs)
                        progress_bar.progress(min(progress, 1.0))
                        
                        # 兼容性處理
                        if train_elapsed is None:
                            train_elapsed = 0
                        
                        # 計算統計
                        avg_time = train_elapsed / epoch if epoch > 0 else 0
                        eta_sec = int(avg_time * (total_epochs - epoch)) if epoch > 0 else 0
                        
                        # 更新最佳 Loss
                        if loss < metrics["best_loss"]:
                            metrics["best_loss"] = loss
                        
                        # 顯示詳細資訊
                        status_text.markdown(f"""
                        **🚀 正在訓練神經網路 ({ticker})...**
                        - **進度**: `Epoch {epoch}/{total_epochs}` ({progress*100:.1f}%)
                        - **當前 Loss**: `{loss:.6f}`
                        - **最佳 Loss**: `{metrics["best_loss"]:.6f}`
                        - **速度**: `{avg_time:.3f} s/epoch`
                        - **預估剩餘時間**: `{eta_sec} 秒`
                        """)
                        
                        # 記錄日誌
                        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}"
                        training_logs.append(log_msg)
                    
                    # 開始訓練
                    with st.spinner(f"正在建立神經網路與數據集..."):
                        start_time = time.time()
                        learn_msg, final_loss = ai_advisor.train_deep_learning_model(
                            ticker, 
                            hist, 
                            progress_callback=on_train_progress
                        )
                        end_time = time.time()
                    
                    # 訓練完成
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    # 顯示結果
                    if final_loss > 0:
                        st.success(learn_msg)
                        st.balloons()
                        
                        # 詳細統計
                        with st.expander("📊 訓練結果統計", expanded=True):
                            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                            with result_col1:
                                st.metric("訓練耗時", f"{end_time - start_time:.1f} 秒")
                            with result_col2:
                                st.metric("最終 Loss", f"{final_loss:.6f}")
                            with result_col3:
                                st.metric("最佳 Loss", f"{metrics['best_loss']:.6f}")
                            with result_col4:
                                st.metric("數據量", f"{len(hist)} 天")
                        
                        # 訓練日誌
                        if training_logs:
                            with st.expander("📋 訓練詳細日誌"):
                                st.text("\n".join(training_logs[-30:]))  # 顯示最後30條
                        
                        # 保存訓練記錄
                        save_training_record({
                            'type': 'stock_model',
                            'ticker': ticker,
                            'stock_name': stock_name,
                            'duration': end_time - start_time,
                            'final_loss': final_loss,
                            'best_loss': metrics['best_loss'],
                            'data_size': len(hist),
                            'timestamp': datetime.now().isoformat()
                        })
                    elif str(final_loss) == "0":
                        if "訓練失敗" in learn_msg or "數據不足" in learn_msg:
                            st.error(learn_msg)
                        else:
                            st.success(learn_msg)
                
            except Exception as e:
                st.error(f"❌ 訓練失敗: {str(e)}")
                import traceback
                with st.expander("錯誤詳情"):
                    st.code(traceback.format_exc())


def render_batch_training(ai_advisor):
    """批量訓練介面"""
    
    st.subheader("📦 批量股票模型訓練")
    st.markdown("""
    自動訓練多支股票的 AI 預測模型
    - ✅ 批量訓練熱門股（快速預設清單）
    - ✅ 智能訓練優良股（AI 自動篩選）
    - ✅ 可中斷的訓練機制
    - ✅ 完整訓練統計報告
    """)
    
    st.divider()
    
    # 兩種批量訓練模式
    col_batch1, col_batch2, col_batch3 = st.columns(3)
    
    # ===== 模式1: 批量訓練熱門股 =====
    with col_batch1:
        st.markdown("### 📦 批量訓練熱門股")
        st.caption("快速訓練預設的熱門股票清單")
        
        popular_stocks = ["0050.TW", "0056.TW", "2330.TW", "2317.TW", "2454.TW"]
        st.info(f"**訓練清單**: {', '.join([s.replace('.TW', '') for s in popular_stocks])}")
        
        if st.button("🚀 開始批量訓練熱門股", type="primary", use_container_width=True):
            st.divider()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_container = st.container()
            
            # 訓練統計
            batch_stats = {"success": 0, "failed": 0, "total": len(popular_stocks)}
            batch_logs = []
            
            for idx, ticker in enumerate(popular_stocks):
                status_text.text(f"🧠 訓練中: {ticker} ({idx+1}/{len(popular_stocks)})")
                
                try:
                    from utils import get_stock_data
                    _, train_hist, _, _ = get_stock_data(ticker)
                    
                    if not train_hist.empty:
                        # 訓練模型（簡化版，不顯示詳細進度）
                        ai_advisor.train_deep_learning_model(ticker, train_hist)
                        batch_stats["success"] += 1
                        batch_logs.append(f"✅ {ticker} - 訓練成功")
                    else:
                        batch_stats["failed"] += 1
                        batch_logs.append(f"❌ {ticker} - 無法獲取數據")
                        
                except Exception as e:
                    batch_stats["failed"] += 1
                    batch_logs.append(f"❌ {ticker} - 訓練失敗: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(popular_stocks))
            
            # 訓練完成
            progress_bar.empty()
            status_text.empty()
            
            # 顯示結果
            with stats_container:
                st.success(f"✅ 批量訓練完成！成功 {batch_stats['success']}/{batch_stats['total']} 支股票")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("總數", batch_stats['total'])
                with result_col2:
                    st.metric("成功", batch_stats['success'], delta=f"+{batch_stats['success']}")
                with result_col3:
                    st.metric("失敗", batch_stats['failed'], delta=f"-{batch_stats['failed']}" if batch_stats['failed'] > 0 else "0")
                
                # 詳細日誌
                with st.expander("📋 訓練日誌"):
                    for log in batch_logs:
                        st.text(log)
                
                # 保存記錄
                save_training_record({
                    'type': 'batch_popular',
                    'total': batch_stats['total'],
                    'success': batch_stats['success'],
                    'failed': batch_stats['failed'],
                    'timestamp': datetime.now().isoformat()
                })
                
                st.balloons()
    
    # ===== 模式2: 智能訓練優良股 =====
    with col_batch2:
        st.markdown("### 🔥 智能訓練優良股")
        st.caption("AI 自動篩選技術指標評分高的股票")
        
        # 初始化狀態
        if 'advisor_batch_active' not in st.session_state:
            st.session_state.advisor_batch_active = False
        
        if not st.session_state.advisor_batch_active:
            if st.button("🧠 啟動智能篩選與訓練", type="primary", use_container_width=True, help="AI 會先進行技術指標篩選，只針對評分較高(趨勢向上)的股票進行深度學習訓練"):
                # 獲取股票清單
                try:
                    df_stocks = pd.read_csv("data/tw_stock_list.csv")
                    
                    if df_stocks.empty:
                        st.error("無法取得股票清單")
                    else:
                        st.session_state.advisor_batch_active = True
                        st.session_state.advisor_batch_tickers = df_stocks['Ticker'].tolist()
                        st.session_state.advisor_batch_idx = 0
                        st.session_state.advisor_batch_stats = {"success": 0, "skip": 0, "error": 0}
                        st.session_state.advisor_batch_logs = []
                        st.rerun()
                except Exception as e:
                    st.error(f"初始化失敗: {str(e)}")
        else:
            st.warning("⚠️ 訓練正在進行中... (可隨時點擊停止)")
            if st.button("🛑 強制停止訓練", use_container_width=True):
                st.session_state.advisor_batch_active = False
                st.warning("訓練已由使用者手動終止")
                st.rerun()
    
    # ===== 模式3: 全都訓練一次(ai_shopping_list_basic.csv) =====
    with col_batch3:
        # 全部優良股訓練區塊
        st.markdown("### 🏆 全部相對優良股票訓練")
        st.caption("AI 將對所有在 ai_shopping_list_basic.csv 檔案中的股票進行深度學習訓練")
        
        # 按鈕啟動訓練
        if st.button("🏁 開始訓練全部優良股", type="primary", use_container_width=True):
            st.divider()
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_container = st.container()

            # [修正] 匯入 get_stock_data
            from utils import get_stock_data
            
            # 讀取優良股清單
            try:
                df = pd.read_csv("data/ai_shopping_list_basic.csv")
                tickers = df['代碼'].tolist() if '代碼' in df.columns else []
                # 將 .TW、.TWO 結尾去除，僅保留數字代碼
                tickers = [str(t).replace('.TW', '').replace('.TWO', '') for t in df['代碼'].tolist()] if '代碼' in df.columns else []
            except Exception as e:
                st.error(f"❌ 無法讀取 ai_shopping_list_basic.csv: {str(e)}")
                tickers = []
            
            # 批量訓練統計初始化
            batch_stats = {"success": 0, "failed": 0, "total": len(tickers)}
            batch_logs = []
            
            # 逐一訓練每支股票
            for idx, ticker in enumerate(tickers):
                status_text.text(f"🧠 訓練中: {ticker} ({idx+1}/{len(tickers)})")
                try:
                    # 取得歷史數據
                    _, train_hist, _, _ = get_stock_data(ticker)
                    if not train_hist.empty:
                        # 執行訓練
                        ai_advisor.train_deep_learning_model(ticker, train_hist)
                        batch_stats["success"] += 1
                        batch_logs.append(f"✅ {ticker} - 訓練成功")
                    else:
                        batch_stats["failed"] += 1
                        batch_logs.append(f"❌ {ticker} - 無法獲取數據")
                except Exception as e:
                    batch_stats["failed"] += 1
                    batch_logs.append(f"❌ {ticker} - 訓練失敗: {str(e)}")
                # 更新進度條
                progress_bar.progress((idx + 1) / len(tickers) if len(tickers) > 0 else 1)
            
            # 訓練結束，清空進度顯示
            progress_bar.empty()
            status_text.empty()
            
            # 顯示訓練結果與統計
            with stats_container:
                st.success(f"✅ 批量訓練完成！成功 {batch_stats['success']}/{batch_stats['total']} 支股票")
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("總數", batch_stats['total'])
                with result_col2:
                    st.metric("成功", batch_stats['success'], delta=f"+{batch_stats['success']}")
                with result_col3:
                    st.metric("失敗", batch_stats['failed'], delta=f"-{batch_stats['failed']}" if batch_stats['failed'] > 0 else "0")
                # 詳細訓練日誌
                with st.expander("📋 訓練日誌"):
                    for log in batch_logs:
                        st.text(log)
                # 保存訓練記錄
                save_training_record({
                    'type': 'batch_all_shopping',
                    'total': batch_stats['total'],
                    'success': batch_stats['success'],
                    'failed': batch_stats['failed'],
                    'timestamp': datetime.now().isoformat()
                })
                st.balloons()

    # ===== 智能訓練執行邏輯 =====
    if st.session_state.get('advisor_batch_active', False):
        st.divider()
        st.markdown("### 🤖 智能訓練進行中...")
        
        all_tickers = st.session_state.advisor_batch_tickers
        idx = st.session_state.advisor_batch_idx
        
        if idx < len(all_tickers):
            ticker = all_tickers[idx]
            
            # 顯示進度
            progress = idx / len(all_tickers)
            st.progress(progress)
            st.text(f"📊 分析中: {ticker} ({idx+1}/{len(all_tickers)})")
            
            try:
                from utils import get_stock_data
                from services.stock_strategy import StockStrategy
                
                # 獲取數據
                _, hist, _, info = get_stock_data(ticker, force_offline=True)
                
                # [關鍵] 數據長度需 >= 60 以便計算季線 (MA60)
                if not hist.empty and len(hist) >= 60:
                    
                    # [修正 1] 實例化類別 (不傳參數)
                    analyzer = StockStrategy()
                    
                    # [修正 2] 獲取分析結果字典
                    analysis_result = analyzer.analyze_stock(ticker, hist)
                    score = analysis_result.get('score', 0)
                    reasons = analysis_result.get('reasons', [])
                    action = analysis_result.get('action', '未知')
                    
                    # [修正 3] 判斷門檻 (>= 5 代表強勢股)
                    if score >= 5:
                        st.info(f"✨ {ticker} 評分 {score} [{action}]，開始訓練...")
                        
                        # 執行訓練
                        ai_advisor.train_deep_learning_model(ticker, hist)
                        
                        st.session_state.advisor_batch_stats["success"] += 1
                        st.session_state.advisor_batch_logs.append(f"✅ {ticker} - 評分 {score}，訓練成功")
                    else:
                        st.session_state.advisor_batch_stats["skip"] += 1
                        # 顯示主要原因
                        main_reason = reasons[0] if reasons else "評分不足"
                        st.session_state.advisor_batch_logs.append(f"⏭️ {ticker} - 評分 {score} ({main_reason})，跳過")
                else:
                    st.session_state.advisor_batch_stats["skip"] += 1
                    st.session_state.advisor_batch_logs.append(f"⏭️ {ticker} - 數據不足 (需 >= 60筆)")
                    
            except Exception as e:
                st.session_state.advisor_batch_stats["error"] += 1
                st.session_state.advisor_batch_logs.append(f"❌ {ticker} - 錯誤: {str(e)}")
            
            # 更新索引並自動重跑
            st.session_state.advisor_batch_idx += 1
            # time.sleep(0.1) 
            st.rerun()
        
        else:
            # 結束邏輯
            st.session_state.advisor_batch_active = False
            stats = st.session_state.advisor_batch_stats
            st.success(f"🎉 訓練完成！成功訓練 {stats['success']} 支優良股")
            
            # 顯示統計與詳細日誌
            cols = st.columns(4)
            cols[0].metric("總分析", len(all_tickers))
            cols[1].metric("成功訓練", stats['success'])
            cols[2].metric("跳過", stats['skip'])
            cols[3].metric("錯誤", stats['error'])
            
            with st.expander("📋 詳細日誌", expanded=True):
                for log in reversed(st.session_state.advisor_batch_logs[-50:]):
                    st.text(log)
            
            save_training_record({
                'type': 'batch_smart',
                'total': len(all_tickers),
                'success': stats['success'],
                'skipped': stats['skip'],
                'error': stats['error'],
                'timestamp': datetime.now().isoformat()
            })
            
            st.balloons()


def render_training_history():
    """訓練歷史與統計"""
    
    st.subheader("📊 訓練歷史與模型統計")
    
    # 讀取訓練記錄
    training_records = load_training_records()
    
    if not training_records:
        st.info("📝 尚無訓練記錄")
        st.caption("💡 記錄存儲於 `data/training_records/` 資料夾，每次訓練生成獨立的 JSON 文件")
        return
    
    # 統計數據
    total_trainings = len(training_records)
    base_model_count = sum(1 for r in training_records if r.get('type') == 'base_model')
    stock_model_count = sum(1 for r in training_records if r.get('type') == 'stock_model')
    batch_count = total_trainings - base_model_count - stock_model_count
    
    # 計算存儲信息
    records_dir = "data/training_records"
    total_size = 0
    file_count = 0
    if os.path.exists(records_dir):
        for filename in os.listdir(records_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(records_dir, filename)
                total_size += os.path.getsize(filepath)
                file_count += 1
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("總訓練次數", total_trainings)
    with col_stat2:
        st.metric("主模型訓練", base_model_count)
    with col_stat3:
        st.metric("個股訓練", stock_model_count)
    with col_stat4:
        st.metric("批量訓練", batch_count)
    
    # 存儲信息
    with st.expander("💾 存儲信息", expanded=False):
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("記錄文件數", f"{file_count} 個")
        with info_col2:
            st.metric("總磁盤使用", f"{total_size / 1024:.1f} KB")
        with info_col3:
            st.metric("平均文件大小", f"{total_size / file_count / 1024:.2f} KB" if file_count > 0 else "0 KB")
        st.caption(f"📁 存儲位置: `{records_dir}/`")
    
    st.divider()
    
    # 訓練記錄表格
    st.subheader("📋 訓練記錄詳情")
    
    # 轉換為 DataFrame
    records_df = []
    for record in reversed(training_records[-50:]):  # 最新50筆
        rec_type = record.get('type', 'unknown')
        
        # 根據類型格式化顯示
        if rec_type == 'base_model':
            type_display = '🚀 主模型'
            name_display = record.get('mode', '-')
            detail_display = '-'
        elif rec_type == 'stock_model':
            ticker = record.get('ticker', 'N/A')
            type_display = f'📈 個股'
            name_display = record.get('stock_name', ticker)
            loss = record.get('final_loss', 0)
            detail_display = f"Loss: {loss:.6f}" if loss else '-'
        elif rec_type in ['batch_popular', 'batch_smart']:
            batch_type = '熱門股' if rec_type == 'batch_popular' else '智能篩選'
            type_display = f'📦 批量'
            name_display = batch_type
            total = record.get('total', 0)
            success = record.get('success', 0)
            detail_display = f"{success}/{total} 成功"
        else:
            type_display = rec_type
            name_display = '-'
            detail_display = '-'
        
        records_df.append({
            '時間': record.get('timestamp', '未知')[:19].replace('T', ' '),
            '類型': type_display,
            '名稱/模式': name_display,
            '詳情': detail_display,
            '耗時(秒)': f"{record.get('duration', 0):.1f}" if 'duration' in record else '-',
            '數據量': record.get('data_size', '-')
        })
    
    if records_df:
        df = pd.DataFrame(records_df)
        st.dataframe(df, use_container_width=True, height=400)
    
    # 記錄管理
    st.divider()
    st.subheader("🔧 記錄管理")
    
    col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)
    
    with col_mgmt1:
        if st.button("🗑️ 清除所有記錄", type="secondary", use_container_width=True):
            st.session_state['confirm_clear'] = True
    
    with col_mgmt2:
        if st.button("📂 開啟記錄資料夾", use_container_width=True):
            import subprocess
            records_dir = os.path.abspath("data/training_records")
            if os.path.exists(records_dir):
                subprocess.Popen(f'explorer "{records_dir}"')
                st.success("✅ 已開啟資料夾")
            else:
                st.warning("⚠️ 資料夾不存在")
    
    with col_mgmt3:
        if training_records:
            oldest = training_records[0].get('timestamp', '未知')[:10]
            newest = training_records[-1].get('timestamp', '未知')[:10]
            st.info(f"📅 記錄範圍\n{oldest} ~ {newest}")
    
    # 確認清除對話框
    if st.session_state.get('confirm_clear', False):
        st.warning("⚠️ 確定要清除所有訓練記錄嗎？此操作無法復原！")
        conf_col1, conf_col2 = st.columns([1, 1])
        with conf_col1:
            if st.button("✅ 確定清除", type="primary", use_container_width=True):
                if clear_training_records():
                    st.success("✅ 所有訓練記錄已清除")
                    st.session_state['confirm_clear'] = False
                    st.rerun()
                else:
                    st.error("❌ 清除失敗")
        with conf_col2:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state['confirm_clear'] = False
                st.rerun()


def check_base_model_status():
    """檢查主模型狀態"""
    model_path = "data/saved_models/base_model.pth"
    
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        size_mb = stat.st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        
        return {
            'exists': True,
            'file_size': f"{size_mb:.1f} MB",
            'last_trained': modified_time.strftime('%Y-%m-%d %H:%M')
        }
    else:
        return {
            'exists': False,
            'file_size': None,
            'last_trained': None
        }


def save_training_record(record):
    """保存訓練記錄（獨立文件，日期編號）"""
    records_dir = "data/training_records"
    
    try:
        os.makedirs(records_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d')
        base_filename = f"training_{date_str}"
        filepath = os.path.join(records_dir, f"{base_filename}.json")
        
        counter = 2
        while os.path.exists(filepath):
            filepath = os.path.join(records_dir, f"{base_filename}_{counter:02d}.json")
            counter += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # [修正] default=str 防止 datetime 物件無法序列化
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        
        return True
    except Exception as e:
        print(f"保存訓練記錄失敗: {e}")
        return False


def load_training_records():
    """讀取訓練記錄（從文件夾讀取所有記錄）"""
    records_dir = "data/training_records"
    
    try:
        if not os.path.exists(records_dir):
            return []
        
        records = []
        for filename in sorted(os.listdir(records_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(records_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                        records.append(record)
                except Exception as e:
                    print(f"讀取 {filename} 失敗: {e}")
                    continue
        
        return records
    except Exception as e:
        print(f"讀取訓練記錄失敗: {e}")
        return []


def clear_training_records():
    """清除訓練記錄（刪除整個文件夾）"""
    records_dir = "data/training_records"
    
    try:
        if os.path.exists(records_dir):
            import shutil
            shutil.rmtree(records_dir)
        return True
    except Exception as e:
        print(f"清除訓練記錄失敗: {e}")
        return False