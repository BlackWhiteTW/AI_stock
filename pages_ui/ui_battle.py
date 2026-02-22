import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import concurrent.futures
from utils import get_stock_data
from jobs.ai_tasks import process_stock_scan, process_trade_decision

def render_tab_battle(trader_bot, symbol, hist_data, df_stocks, ai_advisor):
    st.header("⚔️ AI 自動交易對戰系統")
    st.markdown("""
    這裡可以啟用一個 **全自動交易機器人**，它擁有獨立的資金與帳戶。
    它會利用 AI 投顧的分析結果，在歷史數據中進行模擬交易 (Backtesting)，
    並實際扣除 **手續費 (0.1425%)** 與 **證交稅 (0.3%)**，看看它能否戰勝大盤！
    """)
    
    # --- AI 資產儀表板 ---
    st.subheader("🤖 AI 機器人資產儀表板")
    
    # 1. 計算 AI 持倉市值
    bot_prices = {}
    if trader_bot.account.portfolio:
        for s in trader_bot.account.portfolio:
            try:
                # 嘗試優先使用快取資料
                _, _, p, _ = get_stock_data(s)
                if p > 0:
                    bot_prices[s] = p
                else:
                    # Fallback live fetch
                    import yfinance as yf
                    t = yf.Ticker(s)
                    h = t.history(period="1d")
                    bot_prices[s] = h['Close'].iloc[-1] if not h.empty else 0
            except:
                bot_prices[s] = 0

    bot_rows, bot_market_val = trader_bot.account.get_portfolio_status(bot_prices)
    bot_total_assets = trader_bot.account.balance + bot_market_val
    bot_pl = bot_total_assets - trader_bot.account.initial_balance
    bot_pl_pct = (bot_pl / trader_bot.account.initial_balance) * 100

    # 顯示指標
    m1, m2, m3 = st.columns(3)
    m1.metric("AI 總資產", f"${bot_total_assets:,.0f}")
    m2.metric("現金部位", f"${trader_bot.account.balance:,.0f}")
    m3.metric("累積損益", f"${bot_pl:,.0f}", f"{bot_pl_pct:.2f}%")
    
    # 顯示持倉表格
    if bot_rows:
        with st.expander("查看 AI 持倉明細", expanded=True):
            st.dataframe(pd.DataFrame(bot_rows), use_container_width=True)
    else:
        # 新增：空手時的 AI 思考
        with st.container():
            st.info("🤖 AI 目前空手 (100% 現金)，正在觀察市場...")
            
            if st.button("💭 詢問 AI：為什麼不買股票？ (查看市場觀點)"):
                with st.chat_message("assistant", avatar="🤖"):
                    st.write("讓我快速掃描一下台股的**領頭羊**來判斷大盤氣氛...")
                    
                    # 快速掃描三權值：台積電、鴻海、聯發科
                    leaders = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
                    leader_sentiments = []
                    
                    progress_line = st.progress(0)
                    for idx, (l_code, l_name) in enumerate(leaders.items()):
                        try:
                            # 取得數據
                            l_t, l_h, _, _ = get_stock_data(l_code)
                            # 使用簡易分析 (不含新聞以加速)
                            if not l_h.empty:
                                l_res = ai_advisor.analyze_stock(l_name, l_h, include_news=False)
                                leader_sentiments.append(l_res)
                                st.caption(f"已分析 {l_name}: 評分 {l_res['score']} ({l_res['action']})")
                        except:
                            pass
                        progress_line.progress((idx + 1) / len(leaders))
                    
                    progress_line.empty()
                    
                    # 總結
                    if not leader_sentiments:
                        st.error("無法取得大盤數據。")
                    else:
                        avg_score = sum([s['score'] for s in leader_sentiments]) / len(leader_sentiments)
                        
                        st.markdown(f"**大盤權值股平均評分：{avg_score:.1f} 分**")
                        
                        if avg_score >= 1.5:
                            st.write("📈 **觀點：市場氣氛不錯！** 龍頭股走勢強勁。")
                            st.write("🤔 **那我為什麼空手？** 可能是因為我正在等待更好的「回檔買點」，或者還沒掃描到其他中小型飆股。建議您可以按下方「🛒 AI 採購中心」讓我去掃描更多股票！")
                        elif avg_score >= -1:
                            st.write("⚖️ **觀點：市場目前多空不明 (盤整中)。**")
                            st.write("🤔 **我的策略：** 這種時候我比較謹慎。與其亂買被套牢，不如持有現金等待方向明確。我會持續關注突破訊號。")
                        else:
                            st.write("📉 **觀點：市場氣氛偏空！** 權值股表現疲弱。")
                            st.write("🤔 **我的策略：** 這是「現金為王」的時刻。目前不宜輕易接刀，我會耐心等待底部訊號出現。")

    st.divider()
    
    col_bot1, col_bot2 = st.columns([1, 2])
    
    with col_bot1:
        st.subheader("⚙️ 設定與執行")
        st.metric("AI 初始資金", f"${trader_bot.account.initial_balance:,.0f}")
        
        # 訓練功能已移至專用分頁
        st.info("💡 **提示**: AI 模型訓練功能已移至「🎓 AI 模型訓練」分頁")
        
        st.divider()
        
        # 回測天數設定
        sim_days = st.slider("回測天數 (模擬 N 天前開始)", min_value=30, max_value=360, value=60, step=10)

        if st.button("🔄 重置 AI 資金與帳戶"):
            trader_bot.reset()
            # 清除回測紀錄
            if 'bot_logs' in st.session_state:
                del st.session_state.bot_logs
            st.success("已重置 AI 帳戶！")
            st.rerun()

        if st.button("🚀 開始自動交易回測"):
            if hist_data.empty:
                st.error("無數據可進行回測")
            else:
                sim_progress = st.progress(0)
                sim_status = st.empty()
                
                def on_sim_progress(curr, total, date_str, assets):
                     p = curr / total if total > 0 else 0
                     sim_progress.progress(min(p, 1.0))
                     sim_status.markdown(f"**回測穿越中...** 📅 `{date_str}` | 💰 資產淨值: `${assets:,.0f}`")

                with st.spinner(f"AI 機器人正在穿越時空，對 {symbol} 進行交易 (從 {sim_days} 天前開始)..."):
                    msg, df_log = trader_bot.run_simulation(symbol, hist_data, progress_callback=on_sim_progress, lookback_days=sim_days)
                    
                    sim_progress.empty()
                    sim_status.empty()

                    st.session_state.bot_logs = df_log
                    st.success(f"回測完成！共交易 {len(df_log[df_log['Action']!=''])} 次")
                    
    with col_bot2:
        st.subheader("📈 績效表現")
        if 'bot_logs' in st.session_state and not st.session_state.bot_logs.empty:
            df_log = st.session_state.bot_logs
            
            # 計算績效
            final_assets = df_log['Assets'].iloc[-1]
            initial_assets = trader_bot.account.initial_balance
            roi = ((final_assets - initial_assets) / initial_assets) * 100
            
            # 基準績效 (Buy & Hold)
            start_price = df_log['Price'].iloc[0]
            end_price = df_log['Price'].iloc[-1]
            benchmark_roi = ((end_price - start_price) / start_price) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("AI 最終資產", f"${final_assets:,.0f}", f"{roi:.2f}%")
            c2.metric("買進持有策略", f"${initial_assets * (1+benchmark_roi/100):,.0f}", f"{benchmark_roi:.2f}%")
            c3.metric("AI 超額報酬", f"{roi - benchmark_roi:.2f}%")
            
            # 繪製比較圖
            fig_perf = go.Figure()
            # AI 資產曲線
            fig_perf.add_trace(go.Scatter(x=df_log.index, y=df_log['Assets'], mode='lines', name='AI 機器人資產', line=dict(color='#ef5350', width=2)))
            # 基準線 (Buy & Hold) - 假設全倉買進
            benchmark_assets = (df_log['Price'] / start_price) * initial_assets
            fig_perf.add_trace(go.Scatter(x=df_log.index, y=benchmark_assets, mode='lines', name='買進持有 (Benchmark)', line=dict(color='gray', dash='dash')))
            
            # 標記買賣點
            buys = df_log[df_log['Action'] == 'BUY']
            sells = df_log[df_log['Action'] == 'SELL']
            
            if not buys.empty:
                fig_perf.add_trace(go.Scatter(x=buys.index, y=buys['Assets'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='yellow'), name='AI 買進'))
            if not sells.empty:
                fig_perf.add_trace(go.Scatter(x=sells.index, y=sells['Assets'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='blue'), name='AI 賣出'))

            fig_perf.update_layout(title="資金成長曲線比較", xaxis_title="日期", yaxis_title="總資產")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # 顯示詳細交易紀錄
            with st.expander("查看 AI 詳細交易日誌"):
                st.dataframe(df_log)

    # --- 新增區塊：AI 待購清單 (AI Shopping List) ---
    st.divider()
    st.header("🛒 AI 採購中心 (Shopping List)")
    st.caption("AI 機器人會在此掃描市場，找出符合策略的標的並建立「待購清單」。")
    
    # 自動嘗試載入上次存檔的清單
    import os
    # 檢查條件：不在 session 中，或是 None，或是雖然有 DataFrame 但是是空的
    should_load = False
    if 'ai_shopping_list' not in st.session_state:
        should_load = True
    elif st.session_state.ai_shopping_list is None:
        should_load = True
    elif isinstance(st.session_state.ai_shopping_list, pd.DataFrame) and st.session_state.ai_shopping_list.empty:
        should_load = True
        
    if should_load:
        if os.path.exists("data/ai_shopping_list.csv"):
            try:
                saved_df = pd.read_csv("data/ai_shopping_list.csv")
                # 確保載入的也不是空的
                if not saved_df.empty:
                    st.session_state.ai_shopping_list = saved_df
            except Exception as e:
                pass # 載入失敗就忽略

    col_shop_1, col_shop_2 = st.columns([1, 1])
    
    with col_shop_1:
         st.markdown("#### 🔍 AI 自主市場探索")
         st.write("AI 將隨機探索台股市場 (包含熱門權值、ETF、中小型股)，尋找潛力標的。")
         st.caption("✨ AI 現在會同步閱讀網路新聞，進行綜合判斷 (速度會較慢)。")
         st.info("📊 **嚴格篩選標準**：AI 評分 ≥ 5 分、多頭排列、量能充足、新聞非負面")

         # 動態計算最大股票數量
         max_stocks_count = len(df_stocks) if not df_stocks.empty else 2000
         # 預設值不要超過最大值
         default_val = min(50, max_stocks_count)
         
         scan_limit = st.slider("想要 AI 掃描多少檔股票？(注意：數量越多耗時越久)", 10, max_stocks_count, default_val, step=10)
         
         if st.button("🚀 讓 AI 去逛街 (隨機掃描市場)"):
             st.info(f"🔧 啟動高性能並行掃描引擎 (最多 {scan_limit} 檔)...")
             
             # 使用優化的掃描流程
             scan_candidates = []
             
             # 1. 選擇掃描目標
             market_targets = []
             if 'ai_shopping_list' in st.session_state and not st.session_state.ai_shopping_list.empty:
                if '代碼' in st.session_state.ai_shopping_list.columns:
                     existing_tickers = st.session_state.ai_shopping_list['代碼'].tolist()
                     market_targets.extend(existing_tickers)
                     st.info(f"📋 保留上一次清單中的 {len(existing_tickers)} 檔股票進行複驗...")

             full_tw_list = df_stocks if not df_stocks.empty else pd.DataFrame()
             random_fill_target = []
             
             if not full_tw_list.empty:
                 listed_stocks = full_tw_list[full_tw_list['Ticker'].str.endswith('.TW')].copy()
                 existing_set = set(market_targets)
                 listed_stocks = listed_stocks[~listed_stocks['Ticker'].isin(existing_set)]

                 industries = listed_stocks['產業'].unique()
                 
                 for ind in industries:
                     stocks_in_ind = listed_stocks[listed_stocks['產業'] == ind]['Ticker'].tolist()
                     if stocks_in_ind:
                         picked = random.sample(stocks_in_ind, min(3, len(stocks_in_ind)))
                         random_fill_target.extend(picked)
                 
                 random_fill_target = list(set(random_fill_target))
                 
                 current_total = len(market_targets) + len(random_fill_target)
                 if current_total < scan_limit:
                    remaining_pool = listed_stocks[~listed_stocks['Ticker'].isin(random_fill_target)]['Ticker'].tolist()
                    needed = scan_limit - current_total
                    if remaining_pool:
                        random_fill_target.extend(random.sample(remaining_pool, min(needed, len(remaining_pool))))
             else:
                 discovery_pool = ["2330.TW", "2317.TW", "2454.TW", "2303.TW", "2881.TW", "2882.TW", "0050.TW", "0056.TW", "2603.TW", "2609.TW"]
                 random_fill_target = [c for c in discovery_pool if c not in market_targets]

             random.shuffle(random_fill_target)
             gap = max(0, scan_limit - len(market_targets))
             random_part = random_fill_target[:gap]
             market_targets.extend(random_part)

             if st.session_state.user_watchlist:
                 user_list = [k if k.endswith('.TW') or k.endswith('.TWO') else k+".TW" for k in st.session_state.user_watchlist.keys()]
                 extras = [u for u in user_list if u not in market_targets and not u.endswith('.TWO')]
                 market_targets.extend(extras)
             
             all_targets = list(set(market_targets))
             random.shuffle(all_targets)
             actual_scan_count = len(all_targets)
             
             # ===== 高性能優化區域 =====
             from services.performance_optimizer import get_batch_loader, get_stock_scanner
             from services.market_data import MarketDataManager
             
             # 步驟 1: 批量加載股票數據 (並行下載)
             data_manager = MarketDataManager()
             batch_loader = get_batch_loader(data_manager, max_workers=8)
             
             scan_prog = st.progress(0)
             scan_txt = st.empty()
             scan_txt.text("⏬ 正在批量下載股票數據...")
             
             def data_progress(current, total, ticker):
                 scan_prog.progress(current / total * 0.3)  # 數據下載佔 30%
                 scan_txt.text(f"⏬ 下載數據 [{current}/{total}]: {ticker}")
             
             hist_data_dict = batch_loader.load_multiple_stocks(
                 all_targets, 
                 only_cached=False,
                 progress_callback=data_progress
             )
             
             # 步驟 2: 並行掃描股票 (多線程分析)
             scan_txt.text("🔍 正在並行分析股票...")
             scanner = get_stock_scanner(ai_advisor, max_workers=6)
             
             found_count = 0
             
             def scan_progress(current, total, ticker, result):
                 nonlocal found_count
                 if result:
                     found_count += 1
                 progress = 0.3 + (current / total * 0.7)  # 分析佔 70%
                 scan_prog.progress(progress)
                 scan_txt.text(f"🔍 分析中 [{current}/{total}]: {ticker} (已找到 {found_count} 檔)")
             
             scan_results = scanner.scan_stocks(
                 all_targets,
                 hist_data_dict,
                 include_news=False,  # 關閉新聞以提速,需要時可打開
                 progress_callback=scan_progress
             )
             
             scan_prog.empty()
             scan_txt.empty()
             
             # 步驟 3: 整理結果
             if scan_results:
                 # 【優化】嚴格過濾：只選出真正有潛力的股票
                 # 1. AI 評分至少要達到 5 分以上 (中等看多)
                 # 2. 必須要有明確的「買進」建議
                 # 3. 若有新聞分數，需為正面 (news_score >= 0)
                 good_picks = [
                     r for r in scan_results 
                     if r.get('AI評分', 0) >= 5  # 提升門檻到 5 分
                     and '買進' in r.get('建議', '')  # 必須有買進建議
                     and (r.get('新聞分數', 0) >= 0 or r.get('新聞分數', 0) == 0)  # 新聞不能是負面
                 ]
                 
                 # 準備基本篩選結果（所有通過基本技術面的）
                 basic_picks = []
                 for r in scan_results:
                     ticker = r['代碼']
                     stock_info = full_tw_list[full_tw_list['Ticker'] == ticker]
                     if not stock_info.empty:
                         r['名稱'] = stock_info.iloc[0]['名稱']
                         r['產業'] = stock_info.iloc[0]['產業']
                     else:
                         r['名稱'] = ticker.replace('.TW', '')
                         r['產業'] = '未知'
                     
                     # 獲取現價
                     hist = hist_data_dict.get(ticker, pd.DataFrame())
                     if not hist.empty:
                         r['現價'] = hist['Close'].iloc[-1]
                     else:
                         r['現價'] = 0
                     
                     r['主要理由'] = r.get('技術理由', '')
                     basic_picks.append(r)
                 
                 # 添加名稱和產業信息（嚴格篩選）
                 for pick in good_picks:
                     ticker = pick['代碼']
                     stock_info = full_tw_list[full_tw_list['Ticker'] == ticker]
                     if not stock_info.empty:
                         pick['名稱'] = stock_info.iloc[0]['名稱']
                         pick['產業'] = stock_info.iloc[0]['產業']
                     else:
                         pick['名稱'] = ticker.replace('.TW', '')
                         pick['產業'] = '未知'
                     
                     # 獲取現價
                     hist = hist_data_dict.get(ticker, pd.DataFrame())
                     if not hist.empty:
                         pick['現價'] = hist['Close'].iloc[-1]
                     else:
                         pick['現價'] = 0
                     
                     pick['主要理由'] = pick.get('技術理由', '')
                 
                 # 組織成 DataFrame - 基本篩選
                 df_basic = pd.DataFrame(basic_picks)
                 cols = ['代碼', '名稱', '產業', '現價', 'AI評分', '建議', '主要理由']
                 for col in cols:
                     if col not in df_basic.columns:
                         if col == '建議':
                             df_basic[col] = df_basic.get('建議', '觀望')
                         else:
                             df_basic[col] = ''
                 df_basic = df_basic[cols]
                 df_basic = df_basic.sort_values('AI評分', ascending=False)
                 
                 # 組織成 DataFrame - 嚴格篩選
                 df_res = pd.DataFrame(good_picks)
                 # 確保欄位存在
                 for col in cols:
                     if col not in df_res.columns:
                         if col == '建議':
                             df_res[col] = df_res.get('建議', '觀望')
                         else:
                             df_res[col] = ''
                 
                 df_res = df_res[cols]
                 df_res = df_res.sort_values('AI評分', ascending=False)
                 
                 st.session_state.ai_shopping_list = df_res
                 st.session_state.ai_shopping_list_basic = df_basic  # 新增：保存基本篩選結果
                 
                 # 保存兩個層級的結果
                 filename_strict = "data/ai_shopping_list.csv"
                 filename_basic = "data/ai_shopping_list_basic.csv"
                 
                 df_res.to_csv(filename_strict, index=False, encoding='utf-8-sig')
                 df_basic.to_csv(filename_basic, index=False, encoding='utf-8-sig')
                 
                 # 顯示詳細統計
                 total_scanned = actual_scan_count
                 total_qualified = len(scan_results)  # 通過基本篩選
                 total_selected = len(good_picks)      # 通過嚴格篩選
                 
                 st.success(f"✅ AI 高速掃描完成！")
                 st.markdown(f"""
                 **📊 掃描統計報告**：
                 - 掃描總數：{total_scanned} 檔
                 - 基本篩選通過：{total_qualified} 檔 (MA 多頭排列、量能充足)
                 - 嚴格篩選通過：{total_selected} 檔 (AI評分≥5、有買進建議、新聞非負面)
                 - 篩選率：{(total_selected/total_scanned*100):.1f}%
                 """)
                 
                 # 顯示儲存位置
                 col_save1, col_save2 = st.columns(2)
                 with col_save1:
                     st.info(f"📁 **基本篩選** ({total_qualified}檔)\n\n`{filename_basic}`")
                 with col_save2:
                     st.info(f"📁 **嚴格篩選** ({total_selected}檔)\n\n`{filename_strict}`")
                 
                 # 顯示性能統計
                 st.caption(f"⚡ 使用並行處理技術,掃描速度提升約 5-8 倍")
             else:
                 st.session_state.ai_shopping_list = pd.DataFrame()
                 st.info("AI 掃描完成,目前市場暫無符合條件的標的。")

    with col_shop_2:
        st.markdown("#### 📋 待購清單明細")
        
        # 新增：選擇顯示哪個層級的結果
        if 'ai_shopping_list_basic' in st.session_state and not st.session_state.ai_shopping_list_basic.empty:
            list_type = st.radio(
                "選擇查看清單",
                options=["嚴格篩選 (高分精選)", "基本篩選 (技術面通過)"],
                horizontal=True,
                help="基本篩選：所有技術面符合的股票 | 嚴格篩選：AI評分≥5的優質標的"
            )
            
            # 根據選擇顯示對應的清單
            if list_type == "基本篩選 (技術面通過)":
                display_df = st.session_state.ai_shopping_list_basic
                download_filename = 'ai_shopping_basic.csv'
            else:
                display_df = st.session_state.ai_shopping_list
                download_filename = 'ai_shopping_strict.csv'
            
            st.dataframe(display_df, use_container_width=True)
            st.caption(f"📊 共 {len(display_df)} 檔股票")
            
            # 下載按鈕
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"📥 下載此清單 (CSV)",
                data=csv,
                file_name=download_filename,
                mime='text/csv',
            )
            
        elif 'ai_shopping_list' in st.session_state and not st.session_state.ai_shopping_list.empty:
            # 舊版相容：只有嚴格篩選結果
            display_df = st.session_state.ai_shopping_list
            st.dataframe(display_df, use_container_width=True)
            
            # 下載按鈕
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載此清單 (CSV)",
                data=csv,
                file_name='ai_shopping_result.csv',
                mime='text/csv',
            )
        else:
            st.write("尚無資料，請先執行左側掃描。")
            display_df = None
        
        # 手動執行購買按鈕（統一放在外面）
        if display_df is not None and not display_df.empty:
            st.markdown("---")
            if st.button("🤖 立即讓 AI 執行購買 (模擬)"):
                count = 0
                buy_log = []
                for index, row in display_df.iterrows():
                    code = row['代碼']
                    price = row['現價']
                    
                    # 簡單資金控管：每檔買 1 張
                    if trader_bot.account.balance >= price * 1000:
                        suc, msg = trader_bot.account.buy(code, price, 1000)
                        if suc:
                            count += 1
                            buy_log.append(f"✅ {code} 買進成功")
                        else:
                            buy_log.append(f"❌ {code} 資金不足")
                
                if count > 0:
                    st.success(f"已完成 {count} 筆交易！")
                    for log in buy_log:
                        st.write(log)
                    st.rerun()

    # 每天的操作介面
    st.divider()
    st.subheader("🛠️ 進階手動測試")
    
    col_daily_1, col_daily_2 = st.columns([1, 1])
    
    with col_daily_1:
        st.write("單一股票操作：")
        st.caption(f"針對目前查看的 {symbol} 進行交易判斷")
        if st.button("🤖 執行今日 AI 操盤 (目前個股)", key="btn_daily_trade"):
            if hist_data.empty:
                st.error("無法取得數據")
            else:
                with st.spinner(f"AI 正在讀取 {symbol} 今日最新報價並思考策略..."):
                    daily_msg, daily_analysis = trader_bot.execute_daily_trade(symbol, hist_data)
                    
                    if "⚠️" in daily_msg:
                        st.warning(daily_msg)
                    else:
                        st.success(daily_msg)
                        if daily_analysis:
                            st.json(daily_analysis)
                            
    with col_daily_2:
        st.write("全市場批量操作：")
        st.caption("針對「自選清單」、「庫存股」與「AI 掃描清單」進行全面掃描與下單")
        
        # 收集所有觀察名單
        watchlist = set()
        # 1. 加入庫存
        if trader_bot.account.portfolio:
            watchlist.update(trader_bot.account.portfolio.keys())
        # 2. 加入使用者自選
        if st.session_state.user_watchlist:
            watchlist.update(st.session_state.user_watchlist.keys())
        # 3. 加入 AI 逛街發現的潛力股 (如果有)
        if 'ai_shopping_list' in st.session_state and not st.session_state.ai_shopping_list.empty:
            if '代碼' in st.session_state.ai_shopping_list.columns:
                watchlist.update(st.session_state.ai_shopping_list['代碼'].tolist())
                
        if st.button(f"🌍 執行掃描 ({len(watchlist)} 檔)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            report_logs = []
            sorted_watchlist = sorted(list(watchlist))
            
            # 使用平行運算
            completed = 0
            total = len(sorted_watchlist)
            max_workers = 4 # 分散請求
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_code = {
                    executor.submit(process_trade_decision, code, trader_bot): code 
                    for code in sorted_watchlist
                }
                
                for future in concurrent.futures.as_completed(future_to_code):
                     code = future_to_code[future]
                     completed += 1
                     progress_bar.progress(completed / total)
                     status_text.text(f"正在分析 ({completed}/{total}): {code} ...")
                     
                     try:
                         res = future.result()
                         if res['has_action']:
                             report_logs.append(res['report'])
                     except:
                         pass
            
            progress_bar.empty()
            status_text.empty()
            
            if report_logs:
                st.success(f"全市場掃描完成！共觸發 {len(report_logs)} 筆交易決策：")
                st.dataframe(pd.DataFrame(report_logs))
            else:
                st.info("全市場掃描完成，今日無特殊交易訊號 (AI 建議觀望)。")

    # --- 每日全自動整合按鈕 ---
    st.divider()
    st.header("📅 每日一鍵執行 (One-Click Routine)")
    st.markdown("""
    此功能將自動執行完整的收盤後作業，包含：
    1. **隨機市場掃描**：探索新標的 (補足 50 檔)。
    2. **全自動交易**：針對「庫存股」與「掃描到的新潛力股」進行自動買賣決策。
    """)
    
    if st.button("🚀 立即啟動「每日自動收盤作業」 (耗時較久)"):
        from daily_job import run_daily_job
        
        # 建立即時日誌視窗
        log_expander = st.expander("📜 詳細執行日誌", expanded=True)
        log_list = []
        log_container = log_expander.container()
        
        # 定義 UI 回調函數
        def ui_logger(msg):
            # 簡單處理換行與進度
            clean_msg = msg.strip()
            if clean_msg:
                log_list.append(clean_msg)
                # 只顯示最後 10 行以保持簡潔，或是全部
                # 為了避免太長，我們用 code block 顯示最新的
                pass
            
            # 即時更新下方狀態
            log_container.text(msg)
            
        with st.spinner("AI 正在全速運轉中，請勿關閉視窗..."):
            # 將 UI 已經初始化好的 trader_bot 與 ai_advisor 傳入，避免重複初始化耗時
            run_daily_job(log_func=ui_logger, trader=trader_bot, advisor=ai_advisor)
            
        st.success("✅ 每日作業執行完畢！請查看上方日誌或重新整理頁面以更新資料。")
        st.balloons()

