from utils import get_stock_data
import logging

def process_stock_scan(t_code, full_tw_list, ai_advisor):
    """
    單一股票掃描任務 (尋找潛力股)
    :param t_code: 股票代碼
    :param full_tw_list: 股票清單 DataFrame
    :param ai_advisor: AIAdvisor 實例
    """
    t_name = t_code
    t_industry = "未知"
    
    # 嘗試從列表中取得名稱與產業 (Pandas 讀取唯讀數據是 Thread-safe 的)
    if not full_tw_list.empty:
        # 優化：避免每次都做 DataFrame query，假設 caller 能傳入 meta
        # 但這裡為了相容性保持原樣
        row = full_tw_list[full_tw_list['Ticker'] == t_code]
        if not row.empty:
            t_name = row.iloc[0]['名稱']
            t_industry = row.iloc[0]['產業']
    
    try:
        # 取得數據 (I/O Bound)
        s_t, s_hist, s_price, _ = get_stock_data(t_code)
        
        if not s_hist.empty:
            # [優化] 第一階段：僅進行技術面快篩 (Disable News for speed)
            # 這樣可以避免對 1600+ 檔股票進行網路爬蟲，大幅縮短掃描時間
            # 原始掃描因為含新聞分析 (I/O Bound)，導致 1小時只能跑 4 檔
            res = ai_advisor.analyze_stock(t_code, s_hist, include_news=False)
            
            # 初步篩選：技術面評分 >= 3 (因為還沒加新聞分，門檻稍微放寬)
            if res['score'] >= 3:
                # [優化] 第二階段：針對入選者進行新聞面深層分析
                # 這時候才開啟網路爬蟲
                res_with_news = ai_advisor.analyze_stock(t_code, s_hist, include_news=True)
                
                # 最終篩選：總分 >= 4 且有明確訊號
                if res_with_news['score'] >= 4 and "買" in res_with_news['action']:
                    return {
                        "found": True,
                        "data": {
                            "代碼": t_code,
                            "名稱": t_name,
                            "產業": t_industry,
                            "現價": s_price,
                            "AI 評分": res_with_news['score'],
                            "建議動作": res_with_news['action'],
                            "主要理由": res_with_news['reasons'][0] if res_with_news['reasons'] else ""
                        },
                        "msg": f"✨ 發現潛力股: {t_name}"
                    }
                else:
                    return {"found": False, "msg": f"已掃描: {t_name} (最終評分不足 {res_with_news['score']})"}
            else:
                 return {"found": False, "msg": f"已掃描: {t_name} (技術面評分不足)"}
    except Exception as e:
        return {"found": False, "msg": f"錯誤 {t_code}: {str(e)}"}
        
    return {"found": False, "msg": f"無數據: {t_name}"}

def process_trade_decision(code, trader_bot):
    """單一交易決策任務 (執行下單)"""
    try:
        t_ticker, t_hist, _, _ = get_stock_data(code)
        if not t_hist.empty:
             msg, analysis = trader_bot.execute_daily_trade(t_ticker, t_hist)
             if analysis: 
                 action = analysis.get('action', '')
                 if "買" in action or "賣" in action:
                     return {
                         "has_action": True,
                         "report": {
                             "股票": t_ticker,
                             "建議": action,
                             "詳情": msg
                         }
                     }
    except Exception as e:
        pass
    return {"has_action": False}
