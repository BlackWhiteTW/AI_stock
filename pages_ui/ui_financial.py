import streamlit as st
import pandas as pd
from utils import get_tw_stock_list
from services.financial_data import get_financial_manager


def render_etf_info(ticker, stock_info):
    """
    顯示 ETF 專屬資訊
    """
    st.title("📊 ETF 基金分析")
    st.info("🏦 此為 ETF（指數型基金），顯示基金相關資訊")
    
    # 基本資訊
    st.subheader("📌 基本資訊")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fund_name = stock_info.get('longName', stock_info.get('shortName', ticker))
        st.metric("基金名稱", fund_name[:30] if len(fund_name) > 30 else fund_name)
    
    with col2:
        fund_family = stock_info.get('fundFamily', 'N/A')
        st.metric("基金公司", fund_family[:30] if len(fund_family) > 30 else fund_family)
    
    with col3:
        net_assets = stock_info.get('totalAssets', stock_info.get('netAssets', 0))
        if net_assets > 0:
            assets_billion = net_assets / 1_000_000_000
            st.metric("淨資產", f"{assets_billion:.2f} 十億")
        else:
            st.metric("淨資產", "N/A")
    
    st.markdown("---")
    
    # 價格與報酬
    st.subheader("💰 價格與報酬")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = stock_info.get('regularMarketPrice', stock_info.get('navPrice', 0))
        prev_close = stock_info.get('previousClose', 0)
        change = current_price - prev_close if prev_close > 0 else 0
        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
        st.metric("目前價格", f"{current_price:.2f}", f"{change_pct:+.2f}%")
    
    with col2:
        nav_price = stock_info.get('navPrice', 0)
        if nav_price > 0:
            st.metric("淨值 (NAV)", f"{nav_price:.2f}")
        else:
            st.metric("淨值 (NAV)", "N/A")
    
    with col3:
        ytd_return = stock_info.get('ytdReturn', 0) * 100
        st.metric("今年報酬率", f"{ytd_return:.2f}%")
    
    with col4:
        three_year_return = stock_info.get('threeYearAverageReturn', 0) * 100
        st.metric("三年平均報酬", f"{three_year_return:.2f}%")
    
    st.markdown("---")
    
    # 配息與費用
    st.subheader("💸 配息與費用")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dividend_yield = stock_info.get('yield', stock_info.get('dividendYield', 0))
        if dividend_yield > 0:
            st.metric("配息率", f"{dividend_yield:.2%}")
        else:
            st.metric("配息率", "N/A")
    
    with col2:
        expense_ratio = stock_info.get('netExpenseRatio', 0)
        if expense_ratio > 0:
            st.metric("費用率", f"{expense_ratio:.2%}")
        else:
            st.metric("費用率", "N/A")
    
    with col3:
        pe_ratio = stock_info.get('trailingPE', 0)
        if pe_ratio > 0:
            st.metric("本益比 (PE)", f"{pe_ratio:.2f}")
        else:
            st.metric("本益比 (PE)", "N/A")
    
    with col4:
        beta = stock_info.get('beta3Year', stock_info.get('beta', 0))
        if beta > 0:
            st.metric("Beta 值", f"{beta:.2f}")
        else:
            st.metric("Beta 值", "N/A")
    
    st.markdown("---")
    
    # 交易資訊
    st.subheader("📈 交易資訊")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        day_low = stock_info.get('regularMarketDayLow', 0)
        day_high = stock_info.get('regularMarketDayHigh', 0)
        if day_low > 0 and day_high > 0:
            st.metric("日內區間", f"{day_low:.2f} - {day_high:.2f}")
        else:
            st.metric("日內區間", "N/A")
    
    with col2:
        week52_low = stock_info.get('fiftyTwoWeekLow', 0)
        week52_high = stock_info.get('fiftyTwoWeekHigh', 0)
        if week52_low > 0 and week52_high > 0:
            st.metric("52週區間", f"{week52_low:.2f} - {week52_high:.2f}")
        else:
            st.metric("52週區間", "N/A")
    
    with col3:
        volume = stock_info.get('regularMarketVolume', 0)
        if volume > 0:
            volume_m = volume / 1_000_000
            st.metric("成交量", f"{volume_m:.2f}M")
        else:
            st.metric("成交量", "N/A")
    
    with col4:
        avg_volume = stock_info.get('averageVolume', 0)
        if avg_volume > 0:
            avg_volume_m = avg_volume / 1_000_000
            st.metric("平均成交量", f"{avg_volume_m:.2f}M")
        else:
            st.metric("平均成交量", "N/A")
    
    st.markdown("---")
    
    # 長期績效
    st.subheader("📊 長期績效")
    col1, col2 = st.columns(2)
    
    with col1:
        five_year_return = stock_info.get('fiveYearAverageReturn', 0) * 100
        if five_year_return > 0:
            st.metric("五年平均報酬", f"{five_year_return:.2f}%")
        else:
            st.info("五年平均報酬：資料不足")
    
    with col2:
        three_month_return = stock_info.get('trailingThreeMonthReturns', 0) * 100
        if three_month_return != 0:
            st.metric("近三月報酬", f"{three_month_return:.2f}%")
        else:
            st.info("近三月報酬：資料不足")


def render_financial_page():
    """
    財務報表分析頁面
    顯示股票的財務健康度、財務報表、財務比率等資訊
    """
    st.title("📊 財務報表分析")
    st.markdown("---")
    
    # 🔹 股票選擇器
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_df = get_tw_stock_list()
        if not stock_df.empty:
            # 🔄 從 session_state 讀取側邊欄選擇的股票
            default_ticker = st.session_state.get('selected_ticker', '2330.TW')
            
            # 找到對應的 Display 名稱作為預設值
            default_index = 0
            if default_ticker:
                matching_rows = stock_df[stock_df['Ticker'] == default_ticker]
                if not matching_rows.empty:
                    default_display = matching_rows.iloc[0]['Display']
                    try:
                        default_index = stock_df['Display'].tolist().index(default_display)
                    except ValueError:
                        pass
            
            selected_display = st.selectbox(
                "選擇股票",
                options=stock_df['Display'].tolist(),
                index=default_index,
                key="financial_stock_selector"
            )
            ticker = stock_df[stock_df['Display'] == selected_display]['Ticker'].values[0]
            # 🔄 當用戶在財務頁面切換股票時，也更新 session_state
            st.session_state['selected_ticker'] = ticker
        else:
            ticker = st.text_input("請輸入股票代碼（如 2330.TW）", value="2330.TW")
    
    with col2:
        force_update = st.button("🔄 強制更新", help="重新下載最新的財務報表數據")
    
    if not ticker:
        st.info("請選擇一支股票以查看財務報表")
        return
    
    st.markdown("---")
    
    # 🔹 獲取財務數據
    financial_mgr = get_financial_manager()
    
    with st.spinner("正在載入財務數據..."):
        # 先獲取基本資訊，檢查是否為 ETF
        financial_data = financial_mgr.get_financial_statements(ticker, force_update=force_update)
        
        if not financial_data:
            st.error("❌ 無法獲取股票資訊")
            st.info("💡 可能原因：\n- 該股票代碼不存在\n- yfinance 無法獲取該股票的數據\n- 網路連線問題")
            return
        
        stock_info = financial_data.get('info', {})
        quote_type = stock_info.get('quoteType', 'EQUITY')
        is_etf = quote_type == 'ETF'
        
        # 🔹 如果是 ETF，顯示 ETF 專屬資訊
        if is_etf:
            render_etf_info(ticker, stock_info)
            return
        
        # 🔹 一般股票：獲取財務摘要
        summary = financial_mgr.get_financial_summary(ticker)
        
        if summary.get('status') == 'no_data':
            st.error(f"❌ {summary.get('message', '無法獲取財務數據')}")
            st.info("💡 可能原因：\n- 該股票代碼不存在\n- yfinance 無法獲取該股票的財務數據\n- 網路連線問題")
            return
    
    # 🔹 顯示財務健康度評分
    st.subheader("📈 財務健康度評分")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = summary.get('health_score', 50)
        rating = summary.get('rating', 'B 中等')
        
        # 根據評分顯示不同顏色
        if score >= 80:
            color = "🟢"
        elif score >= 60:
            color = "🟡"
        else:
            color = "🔴"
        
        st.metric(
            label="健康度評分",
            value=f"{score} 分",
            delta=rating,
            help="綜合考量獲利能力、償債能力、估值吸引力和股息吸引力"
        )
        st.markdown(f"### {color} {rating}")
    
    with col2:
        st.info(f"**數據更新時間**\n\n{summary.get('update_time', 'N/A')[:10]}")
    
    with col3:
        st.info(f"**最新財報期間**\n\n{summary.get('latest_period', 'N/A')[:10]}")
    
    st.markdown("---")
    
    # 🔹 七大財務指標類別（新增營收、EPS、持股明細）
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 營收數據",
        "💰 EPS 盈餘",
        "💹 獲利能力",
        "🛡️ 償債能力",
        "💵 估值指標",
        "💎 股息資訊",
        "📊 持股明細"  # 🆕 ETF 成分股分析
    ])
    
    # Tab 1: 營收數據（🆕）
    with tab1:
        st.subheader("📈 月營收與年增率")
        
        try:
            revenue_data = financial_mgr.get_revenue_data(ticker, force_update)
            
            if revenue_data and revenue_data.get('monthly_revenue'):
                monthly_rev = revenue_data.get('monthly_revenue', [])
                
                if monthly_rev:
                    # 顯示最近的營收數據
                    st.success(f"✅ 已載入 {len(monthly_rev)} 筆月營收數據")
                    
                    # 轉換為 DataFrame 以顯示
                    df_revenue = pd.DataFrame(monthly_rev[:12])  # 顯示最近12個月
                    
                    if not df_revenue.empty:
                        st.dataframe(df_revenue, use_container_width=True)
                        
                        # 如果有年增率等欄位，顯示圖表
                        if '年增率' in df_revenue.columns or 'YoY' in df_revenue.columns:
                            st.line_chart(df_revenue.set_index(df_revenue.columns[0]))
                    else:
                        st.info("📊 營收數據格式化中...")
                else:
                    st.info("📊 暫無月營收數據")
            else:
                st.info("📊 Yahoo 財經營收數據不可用，這是正常的（部分股票無此數據）")
                st.caption("💡 營收數據通常每月10日前更新")
        except Exception as e:
            st.warning(f"營收數據載入失敗: {e}")
    
    # Tab 2: EPS 數據（🆕）
    with tab2:
        st.subheader("💰 每股盈餘 (EPS)")
        
        try:
            eps_data = financial_mgr.get_eps_data(ticker, force_update)
            
            if eps_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 如果有季度 EPS
                    if eps_data.get('quarterly_eps'):
                        quarterly_eps = eps_data.get('quarterly_eps', [])
                        st.success(f"✅ 已載入 {len(quarterly_eps)} 筆季度 EPS")
                        
                        df_eps = pd.DataFrame(quarterly_eps[:8])  # 最近8季
                        if not df_eps.empty:
                            st.dataframe(df_eps, use_container_width=True)
                    # 備用：顯示基本 EPS
                    elif eps_data.get('trailing_eps'):
                        st.metric(
                            label="近12個月 EPS",
                            value=f"{eps_data.get('trailing_eps', 0):.2f}",
                            help="過去12個月的每股盈餘"
                        )
                
                with col2:
                    if eps_data.get('forward_eps'):
                        st.metric(
                            label="預估 EPS",
                            value=f"{eps_data.get('forward_eps', 0):.2f}",
                            help="市場預估的未來12個月 EPS"
                        )
                    
                    # 顯示數據來源
                    source = eps_data.get('source', 'Yahoo TW')
                    st.caption(f"📊 數據來源: {source}")
            else:
                st.info("📊 暫無 EPS 數據")
        except Exception as e:
            st.warning(f"EPS 數據載入失敗: {e}")
    
    # Tab 3: 獲利能力
    with tab3:
        st.subheader("獲利能力指標")
        
        profitability = summary.get('profitability', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            roe = profitability.get('roe', 0)
            st.metric(
                label="ROE 股東權益報酬率",
                value=f"{roe:.2f}%",
                help="衡量公司為股東創造報酬的能力\n優秀: >15% | 良好: >10% | 中等: >5%"
            )
            
            # 評級
            if roe > 15:
                st.success("✅ 優秀")
            elif roe > 10:
                st.info("👍 良好")
            elif roe > 5:
                st.warning("⚠️ 中等")
            else:
                st.error("❌ 不佳")
        
        with col2:
            roa = profitability.get('roa', 0)
            st.metric(
                label="ROA 資產報酬率",
                value=f"{roa:.2f}%",
                help="衡量公司運用資產創造利潤的效率\n優秀: >8% | 良好: >5% | 中等: >2%"
            )
            
            if roa > 8:
                st.success("✅ 優秀")
            elif roa > 5:
                st.info("👍 良好")
            elif roa > 2:
                st.warning("⚠️ 中等")
            else:
                st.error("❌ 不佳")
        
        with col3:
            npm = profitability.get('net_profit_margin', 0)
            st.metric(
                label="淨利率",
                value=f"{npm:.2f}%",
                help="每一元營收中有多少轉為淨利\n優秀: >10% | 良好: >5% | 中等: >2%"
            )
            
            if npm > 10:
                st.success("✅ 優秀")
            elif npm > 5:
                st.info("👍 良好")
            elif npm > 2:
                st.warning("⚠️ 中等")
            else:
                st.error("❌ 不佳")
    
    # Tab 4: 償債能力
    with tab4:
        st.subheader("償債能力指標")
        
        solvency = summary.get('solvency', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            debt_ratio = solvency.get('debt_ratio', 0)
            st.metric(
                label="負債比率",
                value=f"{debt_ratio:.2f}%",
                help="總負債占總資產的比例（越低越好）\n優秀: <30% | 良好: <50% | 警戒: >70%"
            )
            
            if debt_ratio < 30:
                st.success("✅ 優秀 - 財務結構穩健")
            elif debt_ratio < 50:
                st.info("👍 良好 - 負債適中")
            elif debt_ratio < 70:
                st.warning("⚠️ 中等 - 需注意負債")
            else:
                st.error("❌ 警戒 - 負債過高")
        
        with col2:
            current_ratio = solvency.get('current_ratio', 0)
            st.metric(
                label="流動比率",
                value=f"{current_ratio:.2f}",
                help="流動資產 / 流動負債（越高越好）\n優秀: >2 | 良好: >1.5 | 警戒: <1"
            )
            
            if current_ratio > 2:
                st.success("✅ 優秀 - 短期償債能力強")
            elif current_ratio > 1.5:
                st.info("👍 良好 - 償債能力充足")
            elif current_ratio > 1:
                st.warning("⚠️ 中等 - 償債能力一般")
            else:
                st.error("❌ 警戒 - 短期償債壓力大")
    
    # Tab 5: 估值指標
    with tab5:
        st.subheader("估值指標")
        
        valuation = summary.get('valuation', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pe = valuation.get('pe_ratio', 0)
            if pe > 0:
                st.metric(
                    label="本益比 (P/E Ratio)",
                    value=f"{pe:.2f}",
                    help="股價 / 每股盈餘\n合理: 15-25 | 便宜: <15 | 昂貴: >30"
                )
                
                if pe < 15:
                    st.success("✅ 估值便宜")
                elif pe < 25:
                    st.info("👍 估值合理")
                elif pe < 35:
                    st.warning("⚠️ 估值偏高")
                else:
                    st.error("❌ 估值昂貴")
            else:
                st.metric(label="本益比 (P/E Ratio)", value="N/A")
                st.info("無數據或公司虧損")
        
        with col2:
            pb = valuation.get('pb_ratio', 0)
            if pb > 0:
                st.metric(
                    label="股價淨值比 (P/B Ratio)",
                    value=f"{pb:.2f}",
                    help="股價 / 每股淨值\n便宜: <1.5 | 合理: 1.5-3 | 昂貴: >5"
                )
                
                if pb < 1.5:
                    st.success("✅ 估值便宜")
                elif pb < 3:
                    st.info("👍 估值合理")
                elif pb < 5:
                    st.warning("⚠️ 估值偏高")
                else:
                    st.error("❌ 估值昂貴")
            else:
                st.metric(label="股價淨值比 (P/B Ratio)", value="N/A")
        
        with col3:
            market_cap = valuation.get('market_cap', 0)
            if market_cap > 0:
                # 轉換為億元
                market_cap_billion = market_cap / 1e8
                st.metric(
                    label="市值",
                    value=f"{market_cap_billion:.0f} 億",
                    help="公司總市值（新台幣）"
                )
                
                if market_cap_billion > 1000:
                    st.info("🏢 大型股")
                elif market_cap_billion > 100:
                    st.info("🏪 中型股")
                else:
                    st.info("🏠 小型股")
            else:
                st.metric(label="市值", value="N/A")
    
    # Tab 6: 股息資訊
    with tab6:
        st.subheader("股息資訊")
        
        dividend = summary.get('dividend', {})
        risk = summary.get('risk', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            dividend_yield = dividend.get('dividend_yield', 0)
            st.metric(
                label="股息殖利率",
                value=f"{dividend_yield:.2f}%",
                help="年度股息 / 股價\n優秀: >5% | 良好: >3% | 中等: >2%"
            )
            
            if dividend_yield > 5:
                st.success("✅ 高股息 - 適合存股")
            elif dividend_yield > 3:
                st.info("👍 中等股息")
            elif dividend_yield > 1:
                st.warning("⚠️ 低股息")
            else:
                st.info("📌 成長股或無配息")
        
        with col2:
            beta = risk.get('beta', 0)
            if beta != 0:
                st.metric(
                    label="Beta 值（風險係數）",
                    value=f"{beta:.2f}",
                    help="衡量股票相對於大盤的波動性\n<1: 較穩定 | 1: 與大盤同步 | >1: 較波動"
                )
                
                if beta < 0.8:
                    st.success("✅ 低波動（防禦型）")
                elif beta < 1.2:
                    st.info("👍 中等波動（平衡型）")
                else:
                    st.warning("⚠️ 高波動（積極型）")
            else:
                st.metric(label="Beta 值", value="N/A")
    
    # Tab 7: 持股明細（🆕 主要用於 ETF）
    with tab7:
        st.subheader("📊 持股明細與成分股分析")
        
        # 判斷是否為 ETF
        clean_ticker = ticker.replace('.TW', '').replace('.TWO', '')
        is_etf = clean_ticker.startswith('00') or 'ETF' in ticker.upper()
        
        if is_etf:
            st.info("🎯 此標的為 ETF，以下顯示持股成分")
        else:
            st.warning("⚠️ 此功能主要適用於 ETF，一般股票可能無持股明細資料")
        
        try:
            # 從 Yahoo TW 爬蟲獲取持股數據
            from services.yahoo_tw_scraper import get_yahoo_scraper
            yahoo_scraper = get_yahoo_scraper()
            holding_data = yahoo_scraper.get_holding(ticker, force_update)
            
            if holding_data and (holding_data.get('top_holdings') or holding_data.get('sector_weights')):
                # 顯示前十大持股
                top_holdings = holding_data.get('top_holdings', [])
                if top_holdings:
                    st.markdown("### 🏆 前十大持股")
                    
                    df_holdings = pd.DataFrame(top_holdings)
                    if not df_holdings.empty:
                        st.dataframe(df_holdings, use_container_width=True)
                        
                        # 如果有比重或權重欄位，顯示圓餅圖
                        weight_col = None
                        for col in df_holdings.columns:
                            if '比重' in col or '權重' in col or '%' in col:
                                weight_col = col
                                break
                        
                        if weight_col and len(df_holdings) > 0:
                            try:
                                import plotly.express as px
                                # 清理百分比符號並轉為數值
                                df_plot = df_holdings.copy()
                                df_plot[weight_col] = df_plot[weight_col].astype(str).str.replace('%', '').str.replace(',', '')
                                df_plot[weight_col] = pd.to_numeric(df_plot[weight_col], errors='coerce')
                                
                                # 獲取名稱欄位
                                name_col = df_plot.columns[0]
                                
                                fig = px.pie(
                                    df_plot.head(10), 
                                    values=weight_col, 
                                    names=name_col,
                                    title="前十大持股權重分布"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.caption(f"圓餅圖顯示失敗: {e}")
                    else:
                        st.info("📊 持股數據解析中...")
                else:
                    st.info("📊 暫無前十大持股數據")
                
                st.markdown("---")
                
                # 顯示產業配置
                sector_weights = holding_data.get('sector_weights', [])
                if sector_weights:
                    st.markdown("### 🏭 產業配置")
                    
                    df_sectors = pd.DataFrame(sector_weights)
                    if not df_sectors.empty:
                        st.dataframe(df_sectors, use_container_width=True)
                        
                        # 產業配置圓餅圖
                        weight_col = None
                        for col in df_sectors.columns:
                            if '比重' in col or '權重' in col or '%' in col:
                                weight_col = col
                                break
                        
                        if weight_col:
                            try:
                                import plotly.express as px
                                df_plot = df_sectors.copy()
                                df_plot[weight_col] = df_plot[weight_col].astype(str).str.replace('%', '').str.replace(',', '')
                                df_plot[weight_col] = pd.to_numeric(df_plot[weight_col], errors='coerce')
                                
                                name_col = df_plot.columns[0]
                                
                                fig = px.bar(
                                    df_plot, 
                                    x=name_col, 
                                    y=weight_col,
                                    title="產業配置權重"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.caption(f"圖表顯示失敗: {e}")
                    else:
                        st.info("📊 產業配置數據解析中...")
                else:
                    st.info("📊 暫無產業配置數據")
                
                # 顯示持股摘要
                holdings_summary = holding_data.get('holdings_summary', {})
                if holdings_summary:
                    st.markdown("### 📋 持股摘要")
                    if holdings_summary.get('raw_text'):
                        st.text(holdings_summary['raw_text'])
                
                # 顯示數據來源
                st.caption(f"📊 數據來源: Yahoo 財經台灣站 | 更新時間: {holding_data.get('update_time', 'N/A')[:19]}")
            else:
                st.info("📊 暫無持股明細數據")
                st.caption("💡 提示：")
                st.caption("  • 此功能主要適用於 ETF")
                st.caption("  • 一般股票通常無持股明細")
                st.caption("  • 可嘗試點擊「🔄 強制更新」重新抓取")
        except Exception as e:
            st.warning(f"持股明細載入失敗: {e}")
            st.caption("💡 可能原因：")
            st.caption("  • 該標的無持股明細資料")
            st.caption("  • 網頁結構變化")
            st.caption("  • 網路連線問題")
    
    st.markdown("---")
    
    # 🔹 詳細財務報表（可選展開）
    with st.expander("📋 查看詳細財務報表"):
        financial_data = financial_mgr.get_financial_statements(ticker, force_update=force_update)
        
        if financial_data:
            st.subheader("季度損益表")
            income_stmt = financial_data.get('income_statement', {})
            if income_stmt:
                try:
                    df_income = pd.DataFrame(income_stmt)
                    st.dataframe(df_income, use_container_width=True)
                except:
                    st.info("損益表數據格式異常")
            else:
                st.info("無季度損益表數據")
            
            st.markdown("---")
            
            st.subheader("季度資產負債表")
            balance_sheet = financial_data.get('balance_sheet', {})
            if balance_sheet:
                try:
                    df_balance = pd.DataFrame(balance_sheet)
                    st.dataframe(df_balance, use_container_width=True)
                except:
                    st.info("資產負債表數據格式異常")
            else:
                st.info("無季度資產負債表數據")
            
            st.markdown("---")
            
            st.subheader("季度現金流量表")
            cash_flow = financial_data.get('cash_flow', {})
            if cash_flow:
                try:
                    df_cash = pd.DataFrame(cash_flow)
                    st.dataframe(df_cash, use_container_width=True)
                except:
                    st.info("現金流量表數據格式異常")
            else:
                st.info("無季度現金流量表數據")
    
    # 🔹 財務分析建議
    st.markdown("---")
    st.subheader("💡 財務分析建議")
    
    score = summary.get('health_score', 50)
    
    if score >= 80:
        st.success("""
        **優秀的財務體質！**
        - ✅ 公司具有良好的獲利能力和財務結構
        - ✅ 適合作為長期投資標的
        - 💡 建議：持續追蹤財報變化，確保穩定性
        """)
    elif score >= 60:
        st.info("""
        **良好的財務狀況**
        - 👍 公司財務指標整體表現不錯
        - 📊 部分指標有改善空間
        - 💡 建議：注意觀察獲利能力和負債變化
        """)
    elif score >= 40:
        st.warning("""
        **中等財務水平**
        - ⚠️ 公司財務指標有待加強
        - 📉 需要關注獲利能力或償債能力
        - 💡 建議：謹慎評估投資風險，搭配技術分析
        """)
    else:
        st.error("""
        **財務狀況需要關注**
        - ❌ 財務指標表現不佳
        - ⚠️ 投資風險較高
        - 💡 建議：避免重倉，等待基本面改善
        """)
