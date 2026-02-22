import streamlit as st
import pandas as pd
from services.ai_prediction_speedup import predict_with_progress_bar
from services.market_scanner_helper import MarketScannerUI, create_scan_mode_selector
import os
import json
import datetime
import torch
import traceback
import traceback

def render_tab_advisor(ai_advisor, symbol, hist_data, current_price, account, df_stocks):
    st.header("🤖 AI 智能投資顧問")
    
    # --- 硬體加速狀態檢查 ---
    
    device_status = "未知"
    device_color = "gray"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_status = f"🚀 GPU 加速已啟用: {gpu_name}"
        device_color = "green"
    else:
        device_status = "⚠️ 目前使用 CPU 運算 (未偵測到 CUDA)"
        device_color = "orange"
        
    st.markdown(f":{device_color}[**{device_status}**]")
    # -----------------------

    st.markdown("本功能會依據**技術指標 (RSI, MACD, 均線, 布林通道)** 對股票進行綜合評分，並提供每日預測驗證。")
    
    # 建立分頁，將原有的個股/掃描功能放入第一頁，新功能放入第二頁
    tab_analysis, tab_prediction_check = st.tabs(["📊 個股與市場掃描", "✅ 每日預測驗證"])
    
    with tab_analysis:
        col_ai_1, col_ai_2 = st.columns([1, 1])
        
        with col_ai_1:
            st.subheader(f"🎯 目標個股分析: {symbol}")
        
        if not hist_data.empty:
            # 顯示目前使用的策略參數
            current_params = ai_advisor._get_params(symbol)
            if 'name' in current_params:
                st.caption(f"當前模式: {current_params['name']}")
            else:
                st.caption("當前模式: 標準均衡型 (預設)")

            # 分析按鈕
            if st.button("🧠 開始分析此股票", key="btn_analyze"):
                with st.spinner("AI 正在計算技術指標數據..."):
                    result = ai_advisor.analyze_stock(symbol, hist_data)
                    
                    # 顯示結果卡片
                    st.divider()
                    st.metric(label="AI 建議方向", value=result['action'], delta=f"信心分數: {result['score']}")
                    
                    if result['reasons']:
                        st.markdown("**📊 分析理由：**")
                        for r in result['reasons']:
                            st.write(f"- {r}")
                            
                    # 資金建議
                    st.markdown("**💰 資金與部位建議：**")
                    can_buy_qty = int(account.balance // current_price)
                    if can_buy_qty > 0 and result['score'] > 0:
                         st.success(f"目前現金 ${account.balance:,.0f}，最大可買進 **{can_buy_qty}** 股。建議分批佈局。")
                    elif result['score'] < 0:
                        # 檢查持倉
                        if symbol in account.portfolio:
                            qty = account.portfolio[symbol]['qty']
                            st.warning(f"偵測到賣出訊號！您目前持有 **{qty}** 股，建議考慮獲利了結或停損。")
                        else:
                            st.info("目前無持倉，建議保持空手觀望。")
                    else:
                        st.info(f"趨勢不明朗，建議觀望。目前現金 ${account.balance:,.0f}。")

                # --- 自動訓練與預測邏辑 ---
                
                # 檢查是否需要自動訓練 (若今日尚未訓練)
                need_train = False
                model_path = os.path.join(ai_advisor.model_dir, f"{symbol}.pth")
                
                if not os.path.exists(model_path):
                    need_train = True
                else:
                    # 檢查檔案修改日期
                    try:
                        mtime = os.path.getmtime(model_path)
                        last_mod = datetime.datetime.fromtimestamp(mtime).date()
                        today = datetime.datetime.now().date()
                        if last_mod < today:
                            need_train = True
                    except:
                        need_train = True
                
                if need_train:
                    st.markdown("---")
                    st.info(f"💡 正在運用 GPU/CPU 為 **{symbol}** 更新當日的深度學習模型...")
                    with st.spinner(f"🚀 Neural Network Training in progress..."):
                        # 自動訓練，不顯示詳細進度條以保持介面簡潔
                        ai_advisor.train_deep_learning_model(symbol, hist_data)
                
                # 執行預測
                pred_price = ai_advisor.predict_next_price(symbol, hist_data)
                
                if pred_price:
                    curr_price = hist_data['Close'].iloc[-1]
                    diff = pred_price - curr_price
                    pct_change = (diff / curr_price) * 100
                    
                    st.markdown("---")
                    st.subheader(f"🔮 {symbol} 股價預測 (24H)")
                    
                    col_p1, col_p2, col_p3 = st.columns(3)
                    col_p1.metric("目前價格", f"{curr_price:.2f}")
                    col_p2.metric("AI 預測價格", f"{pred_price:.2f}", f"{diff:+.2f} ({pct_change:+.2f}%)")
                    
                    with col_p3:
                        if pct_change >= 1.0:
                            st.success("🚀 強力看漲")
                        elif pct_change > 0:
                            st.info("📈 偏多看待")
                        elif pct_change <= -1.0:
                            st.error("🩸 強力看跌")
                        else:
                            st.warning("📉 偏空看待")
                    
                    # 🆕 顯示預測理由 (與綜合分析結果一致)
                    st.markdown("---")
                    st.markdown("### 🧠 AI 判斷依據")
                    
                    # 從分析結果中提取理由
                    if 'reasons' in result and result['reasons']:
                        st.markdown("**📊 主要理由分析：**")
                        
                        # 依照重要性分類顯示
                        core_reasons = [r for r in result['reasons'] if '[核心]' in r or 'AI 模型' in r]
                        tech_reasons = [r for r in result['reasons'] if 'RSI' in r or 'MACD' in r or 'MA' in r]
                        volume_reasons = [r for r in result['reasons'] if '成交量' in r or '出量' in r or '量縮' in r]
                        news_reasons = [r for r in result['reasons'] if '新聞' in r]
                        other_reasons = [r for r in result['reasons'] if r not in core_reasons + tech_reasons + volume_reasons + news_reasons]
                        
                        # 使用 expander 分類顯示，默認展開核心理由
                        if core_reasons:
                            with st.expander("🎯 **核心趨勢判斷** (最重要)", expanded=True):
                                for r in core_reasons:
                                    # 根據內容添加顏色標記
                                    if '多頭' in r or '看多' in r or '強力' in r and '看空' not in r:
                                        st.success(f"✓ {r}")
                                    elif '空頭' in r or '看空' in r or '轉弱' in r or '破線' in r:
                                        st.error(f"✗ {r}")
                                    elif '警告' in r or '利空' in r:
                                        st.warning(f"⚠ {r}")
                                    else:
                                        st.info(f"• {r}")
                        
                        if tech_reasons:
                            with st.expander("📈 **技術指標分析**", expanded=True):
                                for r in tech_reasons:
                                    if '過熱' in r or '偏高' in r or '鈍化' in r:
                                        st.warning(f"⚠ {r}")
                                    elif '低檔' in r or '強勢' in r:
                                        st.success(f"✓ {r}")
                                    else:
                                        st.info(f"• {r}")
                        
                        if volume_reasons:
                            with st.expander("📊 **成交量分析**", expanded=False):
                                for r in volume_reasons:
                                    if '出量攻擊' in r or '量縮' in r:
                                        st.success(f"✓ {r}")
                                    elif '爆量長黑' in r or '出貨' in r:
                                        st.error(f"✗ {r}")
                                    else:
                                        st.info(f"• {r}")
                        
                        if news_reasons:
                            with st.expander("📰 **新聞情感分析**", expanded=False):
                                for r in news_reasons:
                                    if '利多' in r or '正面' in r:
                                        st.success(f"✓ {r}")
                                    elif '利空' in r or '負面' in r:
                                        st.error(f"✗ {r}")
                                    else:
                                        st.info(f"• {r}")
                        
                        if other_reasons:
                            with st.expander("ℹ️ **其他資訊**", expanded=False):
                                for r in other_reasons:
                                    st.info(f"• {r}")
                    
                    # 顯示綜合評分
                    st.markdown("---")
                    col_score1, col_score2 = st.columns(2)
                    with col_score1:
                        score = result.get('score', 0)
                        if score >= 7:
                            st.success(f"**綜合評分**: {score} 分 ⭐⭐⭐")
                        elif score >= 5:
                            st.success(f"**綜合評分**: {score} 分 ⭐⭐")
                        elif score >= 2:
                            st.info(f"**綜合評分**: {score} 分 ⭐")
                        elif score <= -5:
                            st.error(f"**綜合評分**: {score} 分 ⚠️⚠️⚠️")
                        elif score <= -2:
                            st.warning(f"**綜合評分**: {score} 分 ⚠️⚠️")
                        else:
                            st.info(f"**綜合評分**: {score} 分")
                    
                    with col_score2:
                        action = result.get('action', '未知')
                        if '強力買進' in action:
                            st.success(f"**最終建議**: {action}")
                        elif '建議買進' in action:
                            st.success(f"**最終建議**: {action}")
                        elif '觀察' in action:
                            st.info(f"**最終建議**: {action}")
                        elif '賣出' in action:
                            st.error(f"**最終建議**: {action}")
                        else:
                            st.info(f"**最終建議**: {action}")
                            
                    # 顯示模型資訊
                    if symbol in ai_advisor.models:
                        loss = ai_advisor.models[symbol].get('last_loss', 0)
                        
                        # [修正] 嘗試取得模型類型 (從 loaded_models 查找，因為 models 僅存 metadata)
                        m_type = "Deep Learning" 
                        if hasattr(ai_advisor, 'service') and symbol in ai_advisor.service.loaded_models:
                            model_obj = ai_advisor.service.loaded_models[symbol]['model']
                            if "GRU" in str(type(model_obj)):
                                m_type = "GRU"
                            else:
                                m_type = "LSTM"

                        st.caption(f"ℹ️ 模型資訊: Loss={loss:.6f} | 架構: {m_type} | 訓練時間: {datetime.datetime.now().strftime('%H:%M')}")
                    
                    # [新增] 模型驗證按鈕
                    if st.button("📊 驗證此模型 (繪製預測走勢圖)", key="btn_verify_chart"):
                        with st.spinner("AI 正在回測過去 60 天的走勢..."):
                            df_backtest = ai_advisor.get_backtest_data(symbol, hist_data, days=60)
                            
                        if df_backtest is not None and not df_backtest.empty:
                            st.success("✅ 回測完成！請檢視下方圖表：")
                            st.caption("藍線: 實際股價 | 紅線: AI 預測值 (若紅線緊貼藍線，代表模型精確)")
                            
                            # 繪製圖表
                            chart_data = df_backtest.set_index('Date')[['Actual', 'Predicted']]
                            st.line_chart(chart_data, color=["#0000FF", "#FF0000"]) # 藍色實際，紅色預測
                        else:
                            st.error("無法產生回測數據 (可能資料不足或模型尚未訓練)")

                # -----------------------

            st.markdown("---")
            
            # --- 訓練功能已移至 Tab 7「AI 模型訓練中心」 ---
            st.info("💡 **提示**: 深度學習模型訓練功能已移至「🎓 AI 模型訓練」分頁，可進行個股訓練和批量訓練")

            # --- 新增: 顯示預測結果 (若模型存在) ---
            if symbol in ai_advisor.models:
                st.markdown("---")
                st.subheader("🔮 AI 深度學習預測結果")
                
                # 取得預測值
                with st.spinner("模型正在運算推論中..."):
                    pred_price = ai_advisor.predict_next_price(symbol, hist_data)
                
                if pred_price:
                    curr_price = hist_data['Close'].iloc[-1]
                    diff = pred_price - curr_price
                    pct_change = (diff / curr_price) * 100
                    
                    # 顏色邏輯
                    color_delta = "normal"
                    if pct_change > 0.5: color_delta = "off" # Streamlit metric delta color logic is auto usually, specific control via delta_color
                    
                    c_p1, c_p2, c_p3 = st.columns(3)
                    c_p1.metric("當前股價", f"{curr_price:.2f}")
                    c_p2.metric("AI 預測下個交易日", f"{pred_price:.2f}", f"{diff:+.2f} ({pct_change:+.2f}%)")
                    
                    with c_p3:
                        if pct_change > 1.5:
                            st.success("🚀 強力看漲")
                        elif pct_change > 0:
                            st.info("📈 微幅上漲")
                        elif pct_change > -1.5:
                            st.warning("📉 微幅修正")
                        else:
                            st.error("🩸 強力看跌")

                    # 顯示模型信心 (用最後的 Loss 參考)
                    loss = ai_advisor.models[symbol]['last_loss']
                    st.caption(f"模型訓練誤差 (Loss): {loss:.6f} (越低越準確)")
                else:
                    st.warning("數據不足，模型無法進行預測")

    with col_ai_2:
        st.subheader("📡 市場掃描雷達")
        st.info("AI 將掃描您的 **自選清單 (Watchlist)**，找出出現 **買入訊號** 的潛力股。")
        
        if st.session_state.user_watchlist:
            # 使用共用的掃描模式選擇器
            include_news, use_cache = create_scan_mode_selector()
            
            if st.button("🚀 啟動掃描 (自選股)"):
                tickers_list = list(st.session_state.user_watchlist.keys())
                
                # 使用優化的掃描輔助類
                scanner_ui = MarketScannerUI(ai_advisor)
                
                try:
                    results, scan_time = scanner_ui.scan_with_progress(
                        tickers=tickers_list,
                        include_news=include_news,
                        use_cache=use_cache,
                        min_score=2,  # 只顯示分數 >= 2 的
                        use_parallel=True,
                        max_workers=6
                    )
                    
                    # 顯示結果
                    scanner_ui.display_results(results, scan_time)
                    
                except Exception as e:
                    st.error(f"掃描失敗: {e}")
                    with st.expander("查看錯誤詳情"):
                        st.code(traceback.format_exc())
        else:
            st.write("👀 您的自選清單是空的，請先在左側欄位加入股票。")

    with tab_prediction_check:
        st.subheader("🕵️‍♀️ 每日 AI 預測驗證中心")
        
        # 狀態橫幅
        st.success("✅ 系統自動化排程已啟用：將於每日 **08:30** 自動執行預測、**14:30** 自動結算與驗證。保持伺服器開啟即可。")

        # 檢查今日狀態
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        has_pred = False
        has_val = False
        today_pred_count = 0
        
        if os.path.exists(ai_advisor.prediction_file):
            try:
                with open(ai_advisor.prediction_file, 'r', encoding='utf-8') as f:
                    recs = json.load(f)
                today_recs = [r for r in recs if r.get('date') == today_str]
                if today_recs:
                    has_pred = True
                    today_pred_count = len(today_recs)
                    # 檢查是否已驗證 (accuracy 不含 '等待')
                    if any("等待" not in r.get('accuracy', '等待') for r in today_recs):
                        has_val = True
            except: pass

        # 手動控制區 (摺疊以節省空間)
        with st.expander("🛠️ 手動補執行 / 測試控制台 (若排程未觸發可手動執行)", expanded=False):
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### ☀️ 08:30 早盤預測")
                if has_pred:
                    st.success(f"✅ 今日已產生 {today_pred_count} 筆預測")
                    if st.button("🔄 強制重新預測", key="btn_force_pred", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("🚀 使用多線程加速預測中...")
                        
                        ticker_list = list(ai_advisor.models.keys())
                        count = predict_with_progress_bar(
                            ai_advisor, 
                            ticker_list, 
                            progress_bar, 
                            status_text
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ 完成! 預測了 {count} 檔股票")
                        st.rerun()
                else:
                    st.warning("⚠️ 今日尚未執行預測")
                    if st.button("🚀 立即執行預測", key="btn_run_pred", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("🚀 使用多線程加速預測中...")
                        
                        ticker_list = list(ai_advisor.models.keys())
                        count = predict_with_progress_bar(
                            ai_advisor, 
                            ticker_list, 
                            progress_bar, 
                            status_text
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ 完成! 預測了 {count} 檔股票")
                        st.rerun()
                        
            with c2:
                st.markdown("#### 🌙 14:30 收盤驗證")
                if has_val:
                    st.success("✅ 今日已完成收盤驗證")
                    if st.button("🔄 強制重新驗證", key="btn_force_val", use_container_width=True):
                        with st.spinner("正在重新比對收盤價..."):
                            updated_count, _ = ai_advisor.validate_afternoon_results()
                        st.rerun()
                else:
                    if has_pred:
                        st.info("ℹ️ 等待收盤後驗證")
                        if st.button("⚖️ 立即執行驗證", key="btn_run_val", use_container_width=True):
                            with st.spinner("正在獲取收盤價並比對..."):
                                updated_count, _ = ai_advisor.validate_afternoon_results()
                            st.rerun()
                    else:
                        st.error("⛔ 需先執行早盤預測才可驗證")
                        st.button("🚫 無法驗證", disabled=True, key="btn_no_val_disabled", use_container_width=True)
        
        st.markdown("---")
        st.subheader("📑 歷史預測紀錄簿")
        
        if os.path.exists(ai_advisor.prediction_file):
            try:
                with open(ai_advisor.prediction_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                
                if records:
                    df_pred = pd.DataFrame(records)
                    
                    # --- [修正開始] 資料對應與格式優化 ---
                    
                    # 1. 確保 ticker 欄位是字串，並移除可能存在的空白
                    df_pred['ticker'] = df_pred['ticker'].astype(str).str.strip()

                    # 2. 準備對照表 (Mapping)
                    stock_map = {}
                    code_map = {}
                    
                    # 檢查 df_stocks 是否存在且有資料 (假設這是你的全域變數)
                    if 'df_stocks' in locals() and not df_stocks.empty:
                        # 確保 df_stocks 的 Ticker 也是字串
                        df_stocks['Ticker_Clean'] = df_stocks['Ticker'].astype(str).str.strip()
                        
                        # 嘗試處理 .TW / .TWO 的差異 (視你的資料來源而定，如果是 yfinance 會有後綴)
                        # 如果預測檔是 "2330"，但股票清單是 "2330.TW"，這裡做正規化
                        df_stocks['Ticker_Clean'] = df_stocks['Ticker_Clean'].str.replace(r'\.TW.*', '', regex=True)

                        # 建立對照字典
                        if '名稱' in df_stocks.columns:
                            stock_map = dict(zip(df_stocks['Ticker_Clean'], df_stocks['名稱']))
                        elif 'Name' in df_stocks.columns:
                            stock_map = dict(zip(df_stocks['Ticker_Clean'], df_stocks['Name']))
                            
                        if '代碼' in df_stocks.columns:
                            code_map = dict(zip(df_stocks['Ticker_Clean'], df_stocks['代碼']))
                        else:
                            # 如果沒有獨立的代碼欄位，就用 Ticker 當代碼
                            code_map = dict(zip(df_stocks['Ticker_Clean'], df_stocks['Ticker_Clean']))

                    # 3. 執行 Mapping (關鍵修正：使用 fillna 避免 None)
                    # 如果找不到名稱，就顯示原本的 ticker
                    df_pred['stock_name'] = df_pred['ticker'].map(stock_map).fillna(df_pred['ticker'])
                    
                    # 如果找不到代碼，就顯示原本的 ticker
                    df_pred['ticker_clean'] = df_pred['ticker'].map(code_map).fillna(df_pred['ticker'])

                    # 4. 排序：日期(新->舊)，誤差(小->大)
                    df_pred = df_pred.sort_values(by=['date', 'ticker'], ascending=[False, True])
                    
                    # 5. [優化] 處理數值欄位的空值顯示 (避免表格出現 None)
                    # 將尚未產生的數值 (None) 轉為 NaN 以便計算，顯示時再處理
                    cols_to_numeric = ['base_price', 'predicted_price', 'actual_price']
                    for col in cols_to_numeric:
                        if col in df_pred.columns:
                            df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')

                    # --- [修正結束] ---

                    # [新增] 日期篩選器
                    all_dates = sorted(df_pred['date'].unique().tolist(), reverse=True)
                    if all_dates:
                        col_filter_1, col_filter_2 = st.columns([1, 3])
                        with col_filter_1:
                            selected_date = st.selectbox("📅 選擇查看日期", all_dates, index=0)
                        
                        # 過濾資料
                        df_view = df_pred[df_pred['date'] == selected_date].copy()
                        
                        with col_filter_2:
                            # 顯示該日簡單統計
                            total_items = len(df_view)
                            # 計算準確度 (有評級且不含 waiting 的)
                            valid_items = df_view[df_view['accuracy'].astype(str).str.contains('⭐|❌', na=False)]
                            high_acc_items = valid_items[valid_items['accuracy'].astype(str).str.contains('⭐⭐⭐', na=False)]
                            
                            acc_rate = 0
                            if len(valid_items) > 0:
                                acc_rate = (len(high_acc_items) / len(valid_items)) * 100
                            
                            st.caption(f"該日共有 {total_items} 筆預測 | 驗證完成: {len(valid_items)} 筆 | 高準確率: **{acc_rate:.1f}%**")
                    else:
                        df_view = df_pred
                    
                    # 格式化顯示 (指定欄位順序：名稱優先)
                    st.dataframe(
                        df_view, 
                        column_order=[
                            "stock_name", "ticker_clean", 
                            "base_price", "predicted_price", "expected_change_pct", 
                            "actual_price", "diff_pct", "accuracy"
                        ],
                        column_config={
                            "stock_name": "股票名稱",
                            "ticker_clean": "代碼",
                            "base_price": st.column_config.NumberColumn("昨日收盤", format="$%.2f"),
                            "predicted_price": st.column_config.NumberColumn("AI 預測價", format="$%.2f"),
                            "expected_change_pct": st.column_config.NumberColumn(
                                "預期漲跌", 
                                format="%.2f%%",
                                help="預測價格相對於昨日收盤的漲跌幅"
                            ),
                            "actual_price": st.column_config.NumberColumn("實際收盤", format="$%.2f"),
                            "diff_pct": st.column_config.NumberColumn("誤差率", format="%.2f%%"),
                            "accuracy": "準確度評級"
                        },
                        use_container_width=True,
                        hide_index=True,
                        height=800 
                    )
                else:
                    st.info("尚無紀錄。")
            except Exception as e:
                st.error(f"讀取紀錄失敗: {e}")
                # 印出詳細錯誤以便除錯
                st.code(traceback.format_exc())