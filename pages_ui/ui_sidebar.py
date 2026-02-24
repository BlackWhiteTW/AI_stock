import streamlit as st
import pandas as pd
from utils import get_tw_stock_list, get_stock_data
from services.search_optimizer import SearchOptimizer

# 初始化搜尋優化器
search_optimizer = SearchOptimizer()

def render_sidebar(account, ai_advisor=None):
    st.sidebar.title("交易操作面板")
    
    # 嘗試載入清單
    try:
        with st.spinner("正在讀取或更新台股清單 (只含上市股票)..."):
            df_stocks = get_tw_stock_list()
    except:
        df_stocks = pd.DataFrame()

    # --- 側邊欄搜尋介面 ---
    st.sidebar.markdown("### 🔍 股票搜尋")
    input_code = "2330.TW" # 系統預設值

    if not df_stocks.empty:
        # [新增] 搜尋模式切換：分類選單 vs 直接輸入 vs 已訓練模型
        modes = ["📂 分類篩選", "🔍 代碼搜尋"]
        if ai_advisor and hasattr(ai_advisor, 'models') and len(ai_advisor.models) > 0:
            modes.append("🧠 已訓練模型")

        search_mode = st.sidebar.radio("搜尋模式", modes, horizontal=True, label_visibility="collapsed")

        if search_mode == "🔍 代碼搜尋":
            raw_code = st.sidebar.text_input("輸入股票代碼 (例: 2330)", "2330")
            input_code = raw_code.strip()
            # 🔄 同步到 session_state
            st.session_state['selected_ticker'] = input_code if input_code else "2330.TW"
        
        elif search_mode == "🧠 已訓練模型":
            trained_list = sorted(list(ai_advisor.models.keys()))
            display_map = {}
            display_list = []
            
            # [優化] 建立快速查詢表 (O(1) lookup)，避免在迴圈中重複 filter dataframe
            ticker_to_display = {}
            if not df_stocks.empty and 'Ticker' in df_stocks.columns:
                if 'Display' in df_stocks.columns:
                    ticker_to_display = dict(zip(df_stocks['Ticker'], df_stocks['Display']))
                elif '名稱' in df_stocks.columns:
                    # 若無 Display 欄位，利用 代碼+名稱 組合
                    tickers = df_stocks['Ticker']
                    names = df_stocks['名稱']
                    codes = df_stocks['代碼'].astype(str)
                    
                    displays = codes + " " + names
                    ticker_to_display = dict(zip(tickers, displays))
            
            for t in trained_list:
                name = ticker_to_display.get(t, t)
                display_list.append(name)
                display_map[name] = t
            
            selected_display = st.sidebar.selectbox("選擇已訓練模型", display_list)
            if selected_display:
                input_code = display_map[selected_display]
                # 🔄 同步到 session_state
                st.session_state['selected_ticker'] = input_code

        else:
            # 1. 產業選單
            if '產業' in df_stocks.columns:
                all_industries = sorted(df_stocks['產業'].astype(str).unique().tolist())
                # 將 "其他" 移到最後
                if "其他" in all_industries:
                    all_industries.remove("其他")
                    all_industries.append("其他")
                    
                selected_ind = st.sidebar.selectbox("📂 選擇產業分類", all_industries, index=0)
                
                # 2. 股票選單 (過濾)
                filtered_stocks = df_stocks[df_stocks['產業'] == selected_ind]
                
                # 確保顯示欄位存在
                if 'Display' not in filtered_stocks.columns:
                     filtered_stocks['Display'] = filtered_stocks['代碼'] + " " + filtered_stocks['名稱']
                
                selected_stock_display = st.sidebar.selectbox(
                    f"📈 選擇股票 ({len(filtered_stocks)} 檔)", 
                    filtered_stocks['Display'].tolist()
                )
                
                if selected_stock_display:
                    # 取得 Ticker
                    row = filtered_stocks[filtered_stocks['Display'] == selected_stock_display].iloc[0]
                    input_code = row['Ticker']
                    # 🔄 同步到 session_state，讓其他頁面可以讀取
                    st.session_state['selected_ticker'] = input_code
                    
                # 提供強制更新按鈕
                if st.sidebar.button("🔄 更新今日清單"):
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.sidebar.error("清單格式異常 (缺少產業欄位)")
                input_code = st.sidebar.text_input("手動輸入代碼 (Fallback)", "2330.TW")
    else:
        st.sidebar.error("無法取得股票清單。")
        input_code = st.sidebar.text_input("手動輸入代碼", "2330.TW")

    # 獲取股票數據
    try:
        symbol, hist_data, current_price, stock_info = get_stock_data(input_code)
        stock_name = stock_info.get('longName', symbol)
        
        st.sidebar.metric(label=f"{stock_name} ({symbol}) 現價", value=f"{current_price}")

        # --- [優化] AI 智慧搜尋快報 (FinAgent Style) ---
        with st.sidebar.expander("🔍 AI 智慧快報", expanded=True):
            try:
                summary = search_optimizer.get_cached_summary(symbol, ai_advisor)
                if summary:
                    # 頂部數據
                    cols = st.columns(2)
                    cols[0].markdown(f"**產業**: {summary['industry'] or '未知'}")
                    cols[1].markdown(f"**情緒**: {summary['sentiment']['score']:.1f}")
                    
                    st.markdown("---")
                    # LLM 真人診斷 (佔據主要空間)
                    st.markdown("**🧠 AI 真人級診斷**")
                    st.info(summary['ai_verdict'])
                    
                    with st.expander("查看原始數據"):
                        st.write(f"📊 **技術面**: {summary['technical']['trend']['daily']} (MFI: {summary['technical']['mfi']:.1f})")
                        st.write(f"💰 **基本面**: 營收成長 {summary['fundamental']['revenue_growth']:.1f}%")
                else:
                    st.write("無法產生智慧快報")
            except Exception as e:
                st.write(f"快報載入中... {e}")

        # 2. 交易下單
        st.sidebar.markdown("---")
        st.sidebar.subheader("下單")
        
        action = st.sidebar.radio("動作", ["買入", "賣出"], horizontal=True)
        qty = st.sidebar.number_input("數量 (股)", min_value=1, value=1000, step=100)
        
        if st.sidebar.button("送出訂單"):
            if current_price > 0:
                if action == "買入":
                    success, msg = account.buy(symbol, current_price, qty)
                    if success:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)
                else:
                    success, msg = account.sell(symbol, current_price, qty)
                    if success:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)
            else:
                st.sidebar.error("無法取得現價，無法交易")
                
        return symbol, hist_data, current_price, stock_info, df_stocks
        
    except Exception as e:
        st.sidebar.error(f"獲取數據失敗: {e}")
        return input_code, pd.DataFrame(), 0, {}, df_stocks
