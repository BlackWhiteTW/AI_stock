import streamlit as st
import pandas as pd
import yfinance as yf
from utils import get_stock_data
from services.market_scanner_helper import MarketScannerUI, SmartTickerSelector
from services.ai_service import AIService
import random
import os
import time


class SmartDiscoveryEngine:
    """智能市場探索引擎 - AI驅動的選股系統，使用動態候選池"""
    
    def __init__(self):
        # 不再使用硬編碼股票池，改用動態載入
        self.cached_pool_path = "data/ai_shopping_list_basic.csv"
        
        # AI 選股策略配置
        self.strategies = {
            "積極成長": {
                "sectors": ["半導體", "光電", "電子"],
                "min_score": 2,
                "prefer_news": True,
                "description": "🚀 AI 尋找高成長潛力的科技股"
            },
            "穩健配置": {
                "sectors": ["半導體", "金融", "鋼鐵", "ETF"],
                "min_score": 1,
                "prefer_news": False,
                "description": "⚖️ AI 平衡選擇各類型優質標的"
            },
            "價值投資": {
                "sectors": ["金融", "鋼鐵", "化學", "紡織", "ETF"],
                "min_score": 1,
                "prefer_news": False,
                "description": "💰 AI 挖掘被低估的價值股"
            },
            "產業輪動": {
                "sectors": "all",
                "min_score": 2,
                "prefer_news": True,
                "description": "🔄 AI 掃描全市場尋找輪動機會"
            },
            "熱門追蹤": {
                "sectors": ["半導體", "光電", "電子", "通信"],
                "min_score": 2,
                "prefer_news": True,
                "description": "🔥 AI 追蹤市場熱門話題股"
            }
        }
    
    def load_candidate_pool(self):
        """從 CSV 載入候選池並建立產業映射"""
        if not os.path.exists(self.cached_pool_path):
            return None
        
        try:
            df = pd.read_csv(self.cached_pool_path)
            return df
        except Exception as e:
            return None
    
    def filter_by_sectors(self, df, target_sectors):
        """根據產業篩選股票"""
        if df is None or df.empty:
            return []
        
        if target_sectors == "all":
            return df['代碼'].tolist()
        
        # 建立產業關鍵字映射
        sector_keywords = {
            "半導體": ["半導體", "IC"],
            "光電": ["光電"],
            "電子": ["電子", "零組件", "通路"],
            "通信": ["通信", "網路"],
            "金融": ["金融", "保險"],
            "鋼鐵": ["鋼鐵"],
            "化學": ["化學"],
            "紡織": ["紡織"],
            "ETF": ["ETF"],
            "傳產": ["水泥", "食品", "塑膠", "紡織", "電機", "造紙", "鋼鐵", "橡膠", "汽車"]
        }
        
        # 篩選符合產業的股票
        filtered = []
        for sector in target_sectors:
            if sector in sector_keywords:
                keywords = sector_keywords[sector]
                for keyword in keywords:
                    matched = df[df['產業'].str.contains(keyword, na=False, case=False)]
                    filtered.extend(matched['代碼'].tolist())
        
        return list(set(filtered))  # 去重
        
        # AI 選股策略配置
        self.strategies = {
            "積極成長": {
                "sectors": ["AI晶片", "IC設計", "AI應用"],
                "min_score": 2,
                "prefer_news": True,
                "description": "🚀 AI 尋找高成長潛力的科技股"
            },
            "穩健配置": {
                "sectors": ["AI晶片", "金融", "傳產龍頭", "高股息ETF"],
                "min_score": 1,
                "prefer_news": False,
                "description": "⚖️ AI 平衡選擇各類型優質標的"
            },
            "價值投資": {
                "sectors": ["金融", "傳產龍頭", "高股息ETF"],
                "min_score": 1,
                "prefer_news": False,
                "description": "💰 AI 挖掘被低估的價值股"
            },
            "產業輪動": {
                "sectors": "all",
                "min_score": 2,
                "prefer_news": True,
                "description": "🔄 AI 掃描全市場尋找輪動機會"
            },
            "熱門追蹤": {
                "sectors": ["AI晶片", "IC設計", "AI應用", "伺服器/雲端"],
                "min_score": 2,
                "prefer_news": True,
                "description": "🔥 AI 追蹤市場熱門話題股"
            }
        }
    
    def ai_select_stocks(
        self,
        ai_advisor,
        strategy_name: str,
        target_count: int = 10,
        progress_callback=None,
        use_cached_pool: bool = True
    ) -> list:
        """
        使用 AI 智能選股
        
        Args:
            ai_advisor: AI 顧問實例
            strategy_name: 策略名稱
            target_count: 目標選股數量
            progress_callback: 進度回調函數
            use_cached_pool: 是否使用已篩選的候選池（ai_shopping_list_basic.csv）
        
        Returns:
            選中的股票清單 [(ticker, score, analysis), ...]
        """
        strategy_config = self.strategies.get(strategy_name)
        if not strategy_config:
            return []
        
        # 🚀 載入候選池
        candidates = []
        use_prefiltered = False
        
        if use_cached_pool:
            df_pool = self.load_candidate_pool()
            
            if df_pool is not None:
                # 根據策略篩選產業
                target_sectors = strategy_config["sectors"]
                candidates = self.filter_by_sectors(df_pool, target_sectors)
                
                use_prefiltered = True
                if progress_callback:
                    source_desc = "候選池全部股票" if target_sectors == "all" else f"{', '.join(target_sectors[:3])}等產業"
                    progress_callback(f"📂 從候選池載入 {len(candidates)} 檔 ({source_desc})", 0.1)
            else:
                if progress_callback:
                    progress_callback("⚠️ 候選池不存在，請先執行市場掃描", 0)
                return []  # 無候選池則返回空
        else:
            # 不使用候選池時，從完整台股清單掃描
            if progress_callback:
                progress_callback("🔍 正在載入完整台股清單...", 0.05)
            
            try:
                tw_list = pd.read_csv("data/tw_stock_list.csv")
                target_sectors = strategy_config["sectors"]
                
                if target_sectors == "all":
                    # 只掃描上市股票（.TW結尾）
                    candidates = tw_list[tw_list['Ticker'].str.endswith('.TW')]['Ticker'].tolist()
                    # 限制數量避免太慢
                    candidates = random.sample(candidates, min(200, len(candidates)))
                else:
                    # 根據產業篩選
                    candidates = self.filter_by_sectors(tw_list, target_sectors)
                
                if progress_callback:
                    progress_callback(f"📂 從完整清單篩選出 {len(candidates)} 檔候選股", 0.1)
                    
            except Exception as e:
                if progress_callback:
                    progress_callback(f"❌ 載入失敗: {str(e)}", 0)
                return []
        
        if not candidates:
            if progress_callback:
                progress_callback("❌ 沒有找到符合條件的候選股", 0)
            return []
        
        if progress_callback:
            progress_callback(f"🔍 AI 開始分析 {len(candidates)} 檔候選股票...", 0.2)
        
        # 3. AI 快速評分篩選（第一輪：技術面快速掃描）
        scored_stocks = []
        total = len(candidates)
        
        for i, ticker in enumerate(candidates):
            if progress_callback:
                progress = 0.2 + (i + 1) / total * 0.3  # 20%-50% 進度
                progress_callback(f"🧠 AI 評估中... ({i+1}/{total})", progress)
            
            try:
                from utils import get_stock_data
                _, hist, _, _ = get_stock_data(ticker)
                
                if len(hist) > 30:
                    # 快速技術面評分（不含新聞，速度快）
                    quick_analysis = ai_advisor.analyze_stock(ticker, hist, include_news=False)
                    score = quick_analysis.get('score', 0)
                    
                    # 符合策略最低分數要求才保留
                    if score >= strategy_config["min_score"]:
                        scored_stocks.append({
                            'ticker': ticker,
                            'score': score,
                            'analysis': quick_analysis
                        })
            except Exception as e:
                continue
        
        # 4. 排序並選出 top 候選（選多一些以便深度分析後仍有足夠數量）
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = scored_stocks[:min(target_count * 2, len(scored_stocks))]
        
        if progress_callback:
            pool_source = "候選池" if use_prefiltered else "全市場"
            progress_callback(f"✨ 從 {pool_source} {total} 檔中篩選出 {len(top_candidates)} 檔優質候選", 0.5)
        
        # 5. 深度分析（第二輪：含新聞分析，精選最終標的）
        final_selections = []
        
        if strategy_config["prefer_news"] and len(top_candidates) > 0:
            for i, candidate in enumerate(top_candidates[:target_count]):
                if progress_callback:
                    progress = 0.5 + (i + 1) / min(target_count, len(top_candidates)) * 0.5
                    progress_callback(f"📰 AI 深度分析（含新聞）... ({i+1}/{min(target_count, len(top_candidates))})", progress)
                
                try:
                    from utils import get_stock_data
                    _, hist, _, _ = get_stock_data(candidate['ticker'])
                    
                    # 深度分析（含新聞）
                    deep_analysis = ai_advisor.analyze_stock(
                        candidate['ticker'],
                        hist,
                        include_news=True
                    )
                    
                    final_selections.append({
                        'ticker': candidate['ticker'],
                        'score': deep_analysis.get('score', candidate['score']),
                        'analysis': deep_analysis
                    })
                except:
                    # 深度分析失敗，使用快速分析結果
                    final_selections.append(candidate)
        else:
            # 不需要新聞分析，直接使用快速分析結果
            final_selections = top_candidates[:target_count]
        
        # 6. 最終排序
        final_selections.sort(key=lambda x: x['score'], reverse=True)
        
        if progress_callback:
            progress_callback(f"🎯 AI 選股完成！精選 {len(final_selections)} 檔優質標的", 1.0)
        
        return final_selections
    
    def get_available_sectors(self) -> list:
        """取得所有可用產業分類"""
        return ["半導體", "光電", "電子", "通信", "金融", "鋼鐵", "化學", "紡織", "ETF", "傳產"]
    
    def get_strategy_description(self, strategy_name: str) -> str:
        """取得策略描述"""
        config = self.strategies.get(strategy_name)
        return config.get("description", "") if config else ""


# 初始化智能探索引擎
@st.cache_resource
def get_discovery_engine():
    return SmartDiscoveryEngine()

def render_tab_asset(account, symbol, current_price, ai_advisor):
    st.subheader("帳戶資產")
    
    # 手動檢查股息按鈕
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        if st.button("💰 檢查並領取股息"):
            with st.spinner("正在查詢股息發放紀錄..."):
                msgs, summary = account.check_dividends()
                if msgs:
                    st.success(summary)
                    for m in msgs:
                        st.info(m)
                else:
                    st.info(summary)

    # 計算即時市值
    portfolio_stock_prices = {}
    if account.portfolio:
        for s in account.portfolio:
            # 簡單優化：如果剛好是目前查看的股票，直接用
            if s == symbol and current_price > 0:
                portfolio_stock_prices[s] = current_price
            else:
                try:
                    t = yf.Ticker(s)
                    h = t.history(period="1d")
                    p = h['Close'].iloc[-1] if not h.empty else 0
                    portfolio_stock_prices[s] = p
                except:
                    portfolio_stock_prices[s] = 0

    rows, total_market_value = account.get_portfolio_status(portfolio_stock_prices)
    total_assets = account.balance + total_market_value
    
    # 顯示 3 個核心指標
    col1, col2, col3 = st.columns(3)
    col1.metric("總資產", f"${total_assets:,.0f}")
    col2.metric("現金餘額", f"${account.balance:,.0f}")
    
    # 總損益計算 (總資產 - 初始資金)
    total_pl = total_assets - account.initial_balance
    total_pl_pct = (total_pl / account.initial_balance) * 100
    col3.metric("總損益", f"${total_pl:,.0f}", f"{total_pl_pct:.2f}%")
    
    st.divider()
    st.subheader("持倉明細")
    if rows:
        df_portfolio = pd.DataFrame(rows)
        st.dataframe(df_portfolio,  use_container_width=True)
    else:
        st.info("目前沒有持倉")

    # --- 新增：AI 每日操盤室 (Tab 3) ---
    st.divider()
    st.header("🤖 AI 投資經理日誌")
    st.caption("AI 將像人類經理一樣，審視您的持倉並瀏覽標普/台股市場新聞，給出今日建議。")

    if st.button("🧠 喚醒 AI 經理 (進行今日市場分析)"):
        report_container = st.container()
        
        # 1. 持倉健診
        with report_container:
            st.subheader("1. 持倉健診與調整")
            
            if not account.portfolio:
                st.write("📋 目前空手，無需檢視庫存。")
            else:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                # 遍歷持倉
                portfolio_tickers = list(account.portfolio.keys())
                for i, p_ticker in enumerate(portfolio_tickers):
                    status_txt.markdown(f"**正在思考...** 分析持股 `{p_ticker}` 的技術面與新聞面...")
                    progress_bar.progress((i)/len(portfolio_tickers))
                    
                    try:
                        # 取得資料 (Use get_stock_data from utils)
                        p_t, p_hist, _, _ = get_stock_data(p_ticker)
                        if len(p_hist) > 30:
                            # 進行分析 (包含新聞)
                            p_analysis = ai_advisor.analyze_stock(p_ticker, p_hist, include_news=True)
                            
                            # 顯示思考結果
                            sentiment_emoji = ai_advisor.news_analyzer.get_sentiment_emoji(p_analysis['score']) if hasattr(ai_advisor, 'news_analyzer') else "" 
                            with st.expander(f"檢查持股：{p_ticker} ({sentiment_emoji})", expanded=True):
                                col_think_1, col_think_2 = st.columns([3, 1])
                                with col_think_1:
                                    st.markdown(f"**AI 判斷：** {p_analysis['action']}")
                                    st.caption("思考邏輯：")
                                    for r in p_analysis['reasons']:
                                        st.write(f"- {r}")
                                with col_think_2:
                                    st.metric("評分", p_analysis['score'])
                                    
                    except Exception as e:
                         st.error(f"分析 {p_ticker} 時出錯: {e}")
                
                progress_bar.progress(1.0)
                status_txt.empty()
                st.success("✅ 持倉檢視完畢")
        
        # 2. 市場機會掃描
        with report_container:
            st.subheader("2. 今日市場機會探索 (AI 智能選股)")
            
            # 🎯 AI 策略選擇
            discovery_engine = get_discovery_engine()
            
            col_strat1, col_strat2, col_strat3 = st.columns([2, 1, 1])
            with col_strat1:
                strategy = st.selectbox(
                    "🤖 選擇 AI 選股策略",
                    options=list(discovery_engine.strategies.keys()),
                    index=1,  # 預設「穩健配置」
                    help="AI 會根據策略在對應產業中智能篩選優質標的"
                )
            with col_strat2:
                target_count = st.slider("目標選股數", 3, 15, 8, step=1)
            with col_strat3:
                import os
                pool_exists = os.path.exists("data/ai_shopping_list_basic.csv")
                use_pool = st.checkbox(
                    "使用候選池",
                    value=pool_exists,
                    help="使用已篩選的 180 檔候選股（更快）",
                    disabled=not pool_exists
                )
                if pool_exists:
                    st.caption("✅ 候選池可用")
                else:
                    st.caption("❌ 候選池不存在")
            
            # 顯示 AI 策略說明
            st.caption(f"📝 {discovery_engine.get_strategy_description(strategy)}")
            
            # 顯示搜索範圍
            strategy_config = discovery_engine.strategies.get(strategy, {})
            search_sectors = strategy_config.get("sectors", [])
            if search_sectors == "all":
                st.info("🔍 AI 搜索範圍：全市場所有產業")
            elif isinstance(search_sectors, list):
                st.info(f"🔍 AI 搜索範圍：{', '.join(search_sectors)}")
            
            st.write("---")
            
            # 🧠 AI 選股進度顯示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, progress):
                status_text.markdown(f"**{message}**")
                progress_bar.progress(progress)
            
            opportunities = []
            
            # 🚀 使用 AI 驅動的選股引擎
            try:
                update_progress("🤖 AI 正在啟動智能選股引擎...", 0)
                
                # AI 智能選股
                selected_stocks = discovery_engine.ai_select_stocks(
                    ai_advisor=ai_advisor,
                    strategy_name=strategy,
                    target_count=target_count,
                    progress_callback=update_progress,
                    use_cached_pool=use_pool  # 使用候選池選項
                )
                
                # 轉換為機會列表
                for stock_data in selected_stocks:
                    opportunities.append((
                        stock_data['ticker'],
                        stock_data['analysis']
                    ))
                
                progress_bar.empty()
                status_text.empty()
                
                if selected_stocks:
                    st.success(f"✨ AI 從候選池中精選出 {len(selected_stocks)} 檔優質標的！")
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"AI 選股過程出錯: {str(e)}")
                
                # 顯示錯誤詳情
                with st.expander("🔍 查看錯誤詳情"):
                    st.code(str(e))
                
                # Fallback: 使用傳統並行掃描
                st.warning("⚠️ 切換到傳統並行掃描模式...")
                
                try:
                    from services.performance_optimizer import get_stock_scanner
                    scanner = get_stock_scanner(ai_advisor, max_workers=8)
                    
                    # 獲取候選清單（從候選池或完整清單）
                    df_pool = discovery_engine.load_candidate_pool()
                    
                    if df_pool is not None:
                        # 使用候選池
                        strategy_config = discovery_engine.strategies.get(strategy, {})
                        search_sectors = strategy_config.get("sectors", [])
                        scan_list = discovery_engine.filter_by_sectors(df_pool, search_sectors)
                    else:
                        # 使用完整台股清單
                        tw_list = pd.read_csv("data/tw_stock_list.csv")
                        scan_list = tw_list[tw_list['Ticker'].str.endswith('.TW')]['Ticker'].tolist()
                        # 限制數量
                        scan_list = random.sample(scan_list, min(100, len(scan_list)))
                    
                    scan_list = list(set(scan_list))[:target_count * 3]  # 擴大候選池
                    
                    # 準備數據
                    scan_progress = st.progress(0)
                    scan_status = st.empty()
                    stock_data_list = []
                    
                    scan_status.markdown("📥 **正在批次下載數據...**")
                    for i, s_code in enumerate(scan_list):
                        scan_progress.progress(i / len(scan_list) * 0.5)
                        try:
                            s_t, s_hist, _, _ = get_stock_data(s_code)
                            if len(s_hist) > 30:
                                stock_data_list.append((s_t, s_hist))
                        except:
                            continue
                    
                    # 並行分析
                    scan_status.markdown(f"🧠 **並行分析 {len(stock_data_list)} 檔股票...**")
                    results = scanner.scan_stocks_parallel(
                        stock_data_list,
                        include_news=True,
                        use_cache=True
                    )
                    
                    # 篩選高分標的
                    min_score = strategy_config.get("min_score", 1)
                    for ticker, analysis in results.items():
                        if analysis and analysis.get('score', 0) >= min_score:
                            opportunities.append((ticker, analysis))
                    
                    # 排序並取前 N 名
                    opportunities.sort(key=lambda x: x[1].get('score', 0), reverse=True)
                    opportunities = opportunities[:target_count]
                    
                    scan_progress.empty()
                    scan_status.empty()
                    
                    st.success(f"✅ 並行掃描完成！從 {len(stock_data_list)} 檔中找到 {len(opportunities)} 檔優質標的")
                    
                except (ImportError, Exception) as fallback_e:
                    # 最後的備用方案：基礎掃描
                    st.warning(f"並行引擎失敗: {fallback_e}，使用基礎掃描...")
                    
                    # 獲取候選清單（從候選池或完整清單）
                    df_pool = discovery_engine.load_candidate_pool()
                    
                    if df_pool is not None:
                        # 使用候選池
                        strategy_config = discovery_engine.strategies.get(strategy, {})
                        search_sectors = strategy_config.get("sectors", [])
                        scan_list = discovery_engine.filter_by_sectors(df_pool, search_sectors)
                        scan_list = scan_list[:target_count * 2]
                    else:
                        # 使用完整台股清單（隨機抽樣）
                        tw_list = pd.read_csv("data/tw_stock_list.csv")
                        all_stocks = tw_list[tw_list['Ticker'].str.endswith('.TW')]['Ticker'].tolist()
                        scan_list = random.sample(all_stocks, min(target_count * 2, len(all_stocks)))
                    
                    scan_progress = st.progress(0)
                    scan_status = st.empty()
                    min_score = strategy_config.get("min_score", 1)
                    
                    for i, s_code in enumerate(scan_list):
                        scan_status.markdown(f"**正在分析...** `{s_code}` ({i+1}/{len(scan_list)})")
                        scan_progress.progress((i + 1) / len(scan_list))
                        
                        try:
                            s_t, s_hist, _, _ = get_stock_data(s_code)
                            if len(s_hist) > 30:
                                s_analysis = ai_advisor.analyze_stock(s_t, s_hist, include_news=True)
                                if s_analysis['score'] >= min_score:
                                    opportunities.append((s_t, s_analysis))
                        except:
                            continue
                    
                    # 排序並取前 N 名
                    opportunities.sort(key=lambda x: x[1].get('score', 0), reverse=True)
                    opportunities = opportunities[:target_count]
                    
                    scan_progress.empty()
                    scan_status.empty()
                    
                    st.success(f"✅ 基礎掃描完成！找到 {len(opportunities)} 檔優質標的")
            
            if opportunities:
                st.markdown("### 💡 發現投資機會！")
                for opp_ticker, opp_data in opportunities:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(f"**我發現了一檔有趣的股票：{opp_ticker}**")
                        st.markdown(f"**建議：** {opp_data['action']} (分數: {opp_data['score']})")
                        st.markdown("**我的理由是：**")
                        for r in opp_data['reasons']:
                            st.write(f"- {r}")
                        
                        # 如果有新聞，特別標註
                        if any("新聞" in r for r in opp_data['reasons']):
                            st.caption("🌐 (我已查閱相關網路新聞)")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write("我看了一圈市場，目前主要熱門股走勢較不明朗，建議今日多觀察少操作。安全的等待也是一種策略！")

    # --- 帳戶管理 ---
    st.divider()
    if st.button("重置模擬帳戶 (清除所有數據)"):
        account.reset_account()
        st.rerun()
