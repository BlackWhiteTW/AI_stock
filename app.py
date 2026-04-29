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

# 導入 UI 模組 (已開始模組化遷移)
from pages_ui.ui_sidebar import render_sidebar
from pages_ui.ui_analysis import render_tab_analysis
from pages_ui.advisor import render_tab_advisor  # 🚀 已遷移至模組化目錄
from pages_ui.ui_battle import render_tab_battle
from pages_ui.ui_financial import render_financial_page
from pages_ui.ui_testing import render_tab_testing  # 🆕 導入實驗室

# 設定頁面 (必須是第一個 st 命令)
st.set_page_config(page_title="AI 股票模擬交易系統", layout="wide")


# --- 背景排程啟動器 (確保只執行一次) ---
@st.cache_resource
def start_daily_scheduler():
    # 使用 threading.Event 來控制執行緒生命週期 (雖然 daemon 執行緒會隨主進程結束，但這樣更安全)
    stop_event = threading.Event()

    def job_loop():
        print(
            "[System] 🚀 背景排程執行緒啟動 (Thread ID: "
            + str(threading.get_ident())
            + ")"
        )
        print("[System] 等待每日 08:30 (預測) 與 14:30 (交易/驗證) 觸發...")

        while not stop_event.is_set():
            try:
                now = datetime.datetime.now()
                is_weekday = now.weekday() < 5

                # --- 任務 A: 晨間開盤預測 (08:30) ---
                if is_weekday and now.hour == 8 and now.minute == 30:
                    print(
                        f"[System] ☀️ 早盤時間到 ({now.strftime('%H:%M:%S')})！開始執行開盤預測..."
                    )
                    try:
                        # 💡 建立獨立的 advisor 實例給背景任務，不要碰 st.session_state
                        bg_advisor = AIAdvisor()
                        bg_advisor.run_morning_predictions()
                    except Exception as e:
                        print(f"[System] ❌ 早盤排程失敗: {e}")
                    time.sleep(61)

                # --- 任務 B: 下午收盤結算 (14:30) ---
                elif is_weekday and now.hour == 14 and now.minute == 30:
                    print(
                        f"[System] 🌙 收盤時間到 ({now.strftime('%H:%M:%S')})！開始執行自動交易與驗證..."
                    )
                    try:
                        run_daily_job()
                    except Exception as e:
                        print(f"[System] ❌ 下午排程失敗: {e}")
                    time.sleep(61)

                # 每 30 秒檢查一次時間 (不需要太頻繁，減少 CPU 負擔)
                time.sleep(30)
            except Exception as e:
                print(f"[System] ⚠️ 排程迴圈異常: {e}")
                time.sleep(60)

    # 建立並啟動守護執行緒 (Daemon Thread)
    t = threading.Thread(target=job_loop, daemon=True, name="DailyScheduler")
    t.start()
    return t


# 啟動排程 (因為有 cache_resource，重新整理網頁不會重複啟動)
start_daily_scheduler()

# --- Session State 初始化 ---
if "account" not in st.session_state:
    st.session_state.account = PaperAccount()
account = st.session_state.account

if "ai_advisor" not in st.session_state:
    st.session_state.ai_advisor = AIAdvisor()
else:
    # 🚀 修正：當類別定義更新時，確保 session_state 內的實例也同步更新
    # 檢查是否具有最新定義的參數 (例如 force)
    import inspect

    sig = inspect.signature(st.session_state.ai_advisor.validate_afternoon_results)
    if "force" not in sig.parameters:
        st.session_state.ai_advisor = AIAdvisor()

ai_advisor = st.session_state.ai_advisor

if "auto_trader" not in st.session_state:
    st.session_state.auto_trader = AutoTrader()
if "auto_trader_v2" not in st.session_state:
    st.session_state.auto_trader_v2 = AutoTraderV2(
        account_file="data/ai_account_v2.json"
    )
trader_bot_v1 = st.session_state.auto_trader
trader_bot_v2 = st.session_state.auto_trader_v2

if "ai_shopping_list" not in st.session_state:
    st.session_state.ai_shopping_list = pd.DataFrame()

# --- 側邊欄渲染 ---
# render_sidebar 會處理搜尋、下單邏輯，並回傳目前選中的股票資訊
symbol, hist_data, current_price, stock_info, df_stocks = render_sidebar(
    account, ai_advisor
)

# --- 主畫面 ---
st.title("股票模擬訓練系統")

# 分頁定義
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "K 線圖與分析",
        "財務報表分析",
        "AI 智能投顧",
        "AI 自動化股票交易",
        "🧪 AI 實驗室",  # 🆕 v6.0 實驗室
    ]
)

# Tab 1: K 線圖與技術分析
with tab1:
    render_tab_analysis(symbol, hist_data, current_price)

# Tab 2: 財務報表分析
with tab2:
    render_financial_page()

# Tab 3: AI 智能投顧
with tab3:
    render_tab_advisor(ai_advisor, symbol, hist_data, current_price, account, df_stocks)

# Tab 4: AI 自動化股票交易
with tab4:
    render_tab_battle(
        trader_bot_v1=trader_bot_v1,
        trader_bot_v2=trader_bot_v2,
        symbol=symbol,
        hist_data=hist_data,
        df_stocks=df_stocks,
        ai_advisor=ai_advisor,
    )

# Tab 5: AI 實驗室 & 診斷
with tab5:
    render_tab_testing()
