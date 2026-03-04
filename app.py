import streamlit as st
import pandas as pd
import threading
import time
import datetime
from core.trader import PaperAccount
from ai_advisor import AIAdvisor
from core.auto_trader import AutoTrader
from core.auto_trader_v2 import AutoTraderV2
from jobs.daily_job import run_daily_job, run_morning_job

# 導入 UI 模組
from pages_ui.ui_sidebar import render_sidebar
from pages_ui.ui_analysis import render_tab_analysis
from pages_ui.ui_asset import render_tab_asset
from pages_ui.ui_history import render_tab_history
from pages_ui.ui_advisor import render_tab_advisor
from pages_ui.ui_battle import render_tab_battle
from pages_ui.ui_financial import render_financial_page
from pages_ui.ui_training import render_tab_training

# 設定頁面 (必須是第一個 st 命令)
st.set_page_config(page_title="AI 股票模擬交易系統", layout="wide")

# --- 背景排程啟動器 (確保只執行一次) ---
@st.cache_resource
def start_daily_scheduler():
    def job_loop():
        print("[System] 背景排程執行緒已啟動，等待每日 08:30 (預測) 與 14:30 (交易/驗證) 觸發...")
        while True:
            now = datetime.datetime.now()
            is_weekday = now.weekday() < 5
            
            # --- 任務 A: 晨間開盤預測 (08:30) ---
            if is_weekday and now.hour == 8 and now.minute == 30:
                print(f"[System] 早盤時間到 ({now})！開始執行開盤預測...")
                try:
                    run_morning_job()
                except Exception as e:
                    print(f"[System] 早盤排程失敗: {e}")
                time.sleep(65) # 避免同分鐘重複觸發

            # --- 任務 B: 下午收盤結算 (14:30) ---
            elif is_weekday and now.hour == 14 and now.minute == 30:
                print(f"[System] 收盤時間到 ({now})！開始執行每日自動交易與驗證...")
                try:
                    run_daily_job()
                except Exception as e:
                    print(f"[System] 下午排程失敗: {e}")
                time.sleep(65)
            
            # 每 20 秒檢查一次時間，確保精準捕捉
            time.sleep(20)

    # 建立並啟動守護執行緒 (Daemon Thread)，隨主程式關閉而結束
    t = threading.Thread(target=job_loop, daemon=True)
    t.start()
    return t

# 啟動排程 (因為有 cache_resource，重新整理網頁不會重複啟動)
start_daily_scheduler()

# --- Session State 初始化 ---
if 'account' not in st.session_state:
    st.session_state.account = PaperAccount()
account = st.session_state.account

if 'ai_advisor' not in st.session_state:
    st.session_state.ai_advisor = AIAdvisor()
ai_advisor = st.session_state.ai_advisor

if 'auto_trader' not in st.session_state:
    st.session_state.auto_trader = AutoTrader()
if 'auto_trader_v2' not in st.session_state:
    st.session_state.auto_trader_v2 = AutoTraderV2(account_file='data/ai_account_v2.json')
trader_bot_v1 = st.session_state.auto_trader
trader_bot_v2 = st.session_state.auto_trader_v2

if 'user_watchlist' not in st.session_state:
    st.session_state.user_watchlist = {}

if 'ai_shopping_list' not in st.session_state:
    st.session_state.ai_shopping_list = pd.DataFrame()

# --- 側邊欄渲染 ---
# render_sidebar 會處理搜尋、下單邏輯，並回傳目前選中的股票資訊
symbol, hist_data, current_price, stock_info, df_stocks = render_sidebar(account, ai_advisor)

# --- 主畫面 ---
st.title("股票模擬訓練系統")

# 分頁定義
# 注意：這裡使用 st.tabs 回傳的物件
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "K 線圖與分析",
    "財務報表分析", # 🆕 v5.1 財務報表整合
    "資產概況",
    "交易紀錄",
    "AI 智能投顧",
    "AI 自動化股票交易",
    "AI 模型訓練" # 🆕 模型訓練專區
])

# Tab 1: K 線圖與技術分析
with tab1:
    render_tab_analysis(symbol, hist_data, current_price)

# Tab 2: 財務報表分析 (🆕 v5.1)
with tab2:
    render_financial_page()

# Tab 3: 資產概況 (包含 AI 投資經理)
with tab3:
    render_tab_asset(account, symbol, current_price, ai_advisor)

# Tab 4: 歷史交易紀錄
with tab4:
    render_tab_history(account)

# Tab 5: AI 智能投顧 (深度學習訓練)
with tab5:
    render_tab_advisor(ai_advisor, symbol, hist_data, current_price, account, df_stocks)

# Tab 6: AI 自動化股票交易 (包含 AI 採購中心/Scanner)
with tab6:
    render_tab_battle(
        trader_bot_v1=trader_bot_v1, 
        trader_bot_v2=trader_bot_v2, 
        symbol=symbol, 
        hist_data=hist_data, 
        df_stocks=df_stocks, 
        ai_advisor=ai_advisor
    )

# Tab 7: AI 模型訓練中心 (🆕 專業訓練管理)
with tab7:
    render_tab_training(ai_advisor)