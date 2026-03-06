import streamlit as st
import pandas as pd

def render_stock_search(df_stocks, ai_advisor):
    """側邊欄搜尋與標的選擇子模組"""
    st.markdown("### 🔍 股票搜尋")
    input_code = "2330.TW"

    if df_stocks.empty:
        return st.sidebar.text_input("手動輸入代碼", "2330.TW")

    modes = ["📂 分類篩選", "🔍 代碼搜尋"]
    if ai_advisor and hasattr(ai_advisor, 'models') and len(ai_advisor.models) > 0:
        modes.append("🧠 已訓練模型")

    search_mode = st.sidebar.radio("搜尋模式", modes, horizontal=True, label_visibility="collapsed")

    if search_mode == "🔍 代碼搜尋":
        raw_code = st.sidebar.text_input("輸入股票代碼", "2330")
        input_code = raw_code.strip()
        if "." not in input_code: input_code += ".TW"
        st.session_state['selected_ticker'] = input_code
        
    elif search_mode == "🧠 已訓練模型":
        trained_list = sorted(list(ai_advisor.models.keys()))
        input_code = st.sidebar.selectbox("選擇已訓練模型", trained_list)
        st.session_state['selected_ticker'] = input_code

    else:
        # 分類篩選邏輯
        all_industries = sorted(df_stocks['產業'].unique().tolist())
        selected_ind = st.sidebar.selectbox("📂 選擇產業", all_industries)
        filtered = df_stocks[df_stocks['產業'] == selected_ind]
        selected_stock = st.sidebar.selectbox(f"📈 選擇股票 ({len(filtered)})", filtered['Display'].tolist())
        input_code = filtered[filtered['Display'] == selected_stock]['Ticker'].values[0]
        st.session_state['selected_ticker'] = input_code

    return input_code
