import pandas as pd
import random
import os
import sys
import time
import concurrent.futures
from datetime import datetime
from core.auto_trader import AutoTrader
from ai_advisor import AIAdvisor
from utils import get_stock_data, get_tw_stock_list
from jobs.market_trainer import MarketTrainingSystem

def daily_model_training():
    """
    每日自動訓練模型
    
    策略：
    - 重要股票（台灣50、熱門ETF）：每日訓練
    - 一般股票：每週訓練一次
    """
    print("\n" + "=" * 80)
    print("🤖 開始每日模型訓練...")
    print("=" * 80)
    
    try:
        trainer = MarketTrainingSystem()
        
        # 訓練策略：重要股票優先，無最大數量限制
        trainer.train_market(
            max_stocks=None,  # 訓練所有需要更新的股票
            priority='important'  # 重要股票優先
        )
        
        print("✅ 每日模型訓練完成！")
        return True
        
    except Exception as e:
        print(f"❌ 模型訓練失敗: {str(e)}")
        return False

def process_scan_task(ticker, sorted_df_stocks, advisor):
    """平行掃描任務"""
    try:
        _, hist, price, _ = get_stock_data(ticker)
        if not hist.empty and len(hist) > 30:
            # 進行全方位分析 (包含新聞)
            res = advisor.analyze_stock(ticker, hist, include_news=True)
            
            # 篩選邏輯：分數 >= 1 且有買進建議
            if res['score'] >= 1 and "買" in res['action']:
                # 查找基本資料 (名稱/產業)
                name = ticker
                industry = "未知"
                row = sorted_df_stocks[sorted_df_stocks['Ticker'] == ticker]
                if not row.empty:
                    name = row.iloc[0]['名稱']
                    industry = row.iloc[0]['產業']
                    
                return {
                    "valid": True,
                    "data": {
                        "代碼": ticker,
                        "名稱": name,
                        "產業": industry,
                        "現價": price,
                        "AI 評分": res['score'],
                        "建議動作": res['action'],
                        "主要理由": res['reasons'][0] if res['reasons'] else ""
                    },
                    "msg": f"✨ 發現: {name}"
                }
    except Exception:
        pass
    return {"valid": False, "msg": ticker}

def process_trade_task(ticker, trader):
    """平行交易決策任務"""
    try:
        _, hist, _, _ = get_stock_data(ticker)
        if not hist.empty:
            msg, analysis = trader.execute_daily_trade(ticker, hist)
            return {"valid": True, "msg": msg, "analysis": analysis, "ticker": ticker}
    except Exception as e:
        return {"valid": False, "msg": f"Error: {e}", "ticker": ticker}
    return {"valid": False, "msg": "No Data", "ticker": ticker}

def run_morning_job(log_func=print):
    """
    [自動化] 開盤前任務 (08:30)
    1. 執行 AI 模型對所有追蹤股票的預測紀錄
    """
    log_func(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ☀️ [Morning Job] 啟動開盤前 AI 預測任務...")
    
    advisor = AIAdvisor()
    try:
        count = advisor.run_morning_predictions()
        log_func(f"✅ 開盤預測完成！已記錄 {count} 檔模型的今日預測價。")
    except Exception as e:
        log_func(f"❌ 開盤預測執行失敗: {e}")

def run_daily_job(log_func=print, trader=None, advisor=None):
    """
    執行每日自動交易排程 (支援多核心平行運算)
    包含: 收盤驗證, 市場掃描, 自動交易
    :param log_func: 用於輸出日誌的函數，預設為 print
    :param trader: 外部傳入的 AutoTrader 下單機器人 (若無則自動建立)
    :param advisor: 外部傳入的 AIAdvisor 顧問 (若無則自動建立)
    """
    log_func(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🌙 [Afternoon Job] 啟動每日自動交易排程...")
    
    # 初始化
    if trader is None:
        log_func("正在初始化 虛擬帳戶...")
        trader = AutoTrader(account_file='data/ai_account.json')
        
    if advisor is None:
        log_func("正在初始化 AI 模型...")
        advisor = AIAdvisor()
        
    # --- 階段零: 驗證今日早盤預測結果 (Validation) ---
    log_func(">>> 正在執行: 驗證今日 AI 預測準確度...")
    try:
        updated_count, _ = advisor.validate_afternoon_results()
        if updated_count > 0:
            log_func(f"✅ 已完成 {updated_count} 筆預測結果的收盤驗證。")
        else:
            log_func("⚠️ 無法驗證今日預測 (可能無預測記錄或無收盤資料)。")
    except Exception as e:
        log_func(f"❌ 驗證過程發生錯誤: {e}")
    
    # 1. 取得台股清單
    log_func("正在讀取股票清單...")
    df_stocks = get_tw_stock_list()
    if df_stocks.empty:
        log_func("❌ 無法取得股票清單，程序終止")
        return

    # 2. 執行隨機市場掃描 (Shopping List Update)
    log_func("\n-------------------------------------------")
    log_func("--- 階段一: 執行隨機市場掃描 (Shopping List) ---")
    log_func("-------------------------------------------")
    
    # 載入舊清單
    current_shopping_list = []
    if os.path.exists("data/ai_shopping_list.csv"):
        try:
            old_df = pd.read_csv("data/ai_shopping_list.csv")
            if not old_df.empty:
                current_shopping_list = old_df['代碼'].tolist()
        except:
            pass
            
    # 設定掃描目標數
    target_count = 50
    candidates = []
    
    # A. 保留舊的 (確保持續追蹤)
    candidates.extend(current_shopping_list)
    log_func(f"保留追蹤清單: {len(candidates)} 檔")

    # B. 隨機補足新標的 (只選上市 .TW)
    if 'Ticker' in df_stocks.columns:
        # 只選 .TW 結尾 (上市)
        listed_stocks = df_stocks[df_stocks['Ticker'].str.endswith('.TW')]['Ticker'].tolist()
        
        # 排除已在候選清單中的
        pool = [s for s in listed_stocks if s not in candidates]
        
        needed = target_count - len(candidates)
        
        # 如果還缺額，進行隨機抽樣
        if needed > 0 and pool:
            # 簡單隨機抽樣
            new_picks = random.sample(pool, min(needed, len(pool)))
            candidates.extend(new_picks)
            log_func(f"隨機新增探索: {len(new_picks)} 檔")
            
    # 去重
    candidates = list(set(candidates))
    new_shopping_results = []
    
    log_func(f"本次預計掃描總數: {len(candidates)} 檔")
    log_func("AI 正在閱讀新聞與分析技術面 (平行加速運算中)...")
    
    # 平行處理設定
    max_workers = 4 # 避免 Rate Limit
    completed = 0
    total = len(candidates)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_scan_task, ticker, df_stocks, advisor): ticker 
            for ticker in candidates
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            try:
                res = future.result()
                
                # 進度顯示
                if log_func == print:
                    sys.stdout.write(f"\r[{completed}/{total}] {res.get('msg', ticker)} ...     ")
                    sys.stdout.flush()
                else:
                    if completed % 5 == 0 or completed == total or res['valid']:
                        log_func(f"[{completed}/{total}] {res.get('msg', ticker)}")
                        
                if res['valid']:
                    new_shopping_results.append(res['data'])
                    
            except Exception as e:
                pass
            
    log_func(f"\n掃描完成！共發現 {len(new_shopping_results)} 檔高評分潛力股。")
    
    # 存檔 ai_shopping_list.csv
    if new_shopping_results:
        df_res = pd.DataFrame(new_shopping_results)
        # 固定欄位順序
        cols = ['代碼', '名稱', '產業', '現價', 'AI 評分', '建議動作', '主要理由']
        # 確保當中有缺漏的欄位補齊
        for c in cols:
            if c not in df_res.columns: df_res[c] = ""
        df_res = df_res[cols]
        
        df_res.to_csv("data/ai_shopping_list.csv", index=False, encoding='utf-8-sig')
        log_func("✅ 已更新購物清單至: data/ai_shopping_list.csv")
    else:
        # 若這次掃描完全沒有好的，可以選擇清空或保留備份。這裡選擇覆蓋為空，代表市場狀況不佳。
        log_func("⚠️ 掃描後無符合條件標的，清空購物清單。")
        pd.DataFrame(columns=['代碼', '名稱', '產業', '現價', 'AI 評分', '建議動作', '主要理由']).to_csv("data/ai_shopping_list.csv", index=False, encoding='utf-8-sig')


    # 3. 執行全自動交易 (針對 庫存 + Shopping List)
    log_func("\n---------------------------------------------------")
    log_func("--- 階段二: 執行自動交易決策 (Portfolio + Shopping) ---")
    log_func("---------------------------------------------------")
    
    trade_targets = set()
    
    # A. 必掃描：庫存股 (檢查是否該賣出)
    if trader.account.portfolio:
        trade_targets.update(trader.account.portfolio.keys())
        log_func(f"加入庫存股檢查: {len(trade_targets)} 檔")
    
    # B. 必掃描：購物清單 (檢查是否該買進)
    if new_shopping_results:
        shop_tickers = [x['代碼'] for x in new_shopping_results]
        trade_targets.update(shop_tickers)
    
    target_list = sorted(list(trade_targets))
    log_func(f"總計執行交易判斷: {len(target_list)} 檔")
    
    executed_count = 0
    t_completed = 0
    t_total = len(target_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_trade_task, ticker, trader): ticker 
            for ticker in target_list
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            t_completed += 1
            ticker = future_to_ticker[future]
            
            try:
                res = future.result()
                
                # 進度 (實際有交易或錯誤才顯示詳細，不然 update 太快)
                if log_func == print:
                    sys.stdout.write(f"\r交易決策計算中: {ticker} ({t_completed}/{t_total})")
                    sys.stdout.flush()
                else:
                    if t_completed % 5 == 0 or t_completed == t_total:
                         log_func(f"進度 ({t_completed}/{t_total})...")
                
                if res['valid']:
                    msg = res['msg']
                    # 檢查是否有實際動作
                    if "成功" in msg:
                        log_func(f"💰 [成交] {ticker}: {msg}")
                        executed_count += 1
                    elif "資金不足" in msg:
                        log_func(f"⚠️ [流標] {ticker}: {msg}")
                else:
                    # log_func(f"❌ 錯誤 {ticker}: {res['msg']}")
                    pass

            except Exception as e:
                log_func(f"❌ 錯誤 {ticker}: {e}")
            
    log_func(f"\n🎉 每日排程執行完畢。共執行 {executed_count} 筆交易。")
    log_func(f"目前 AI 總資產: ${trader.account.get_total_assets():,.0f}")
    
    # 🆕 每日模型訓練
    log_func(f"\n開始每日模型訓練...")
    daily_model_training()
    
    # 稍微暫停讓視窗不要馬上關閉 (如果用視窗執行的話)
    if log_func == print:
        time.sleep(3)

if __name__ == "__main__":
    run_daily_job()