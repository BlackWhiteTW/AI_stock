import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_tab_analysis(symbol, hist_data, current_price):
    if not hist_data.empty:
        st.subheader(f"{symbol} 走勢圖")

        # --- 時間區間選擇 ---
        col_range, col_dummy = st.columns([2, 1])
        with col_range:
            range_options = ["近1週", "近1月", "近1季", "近半年", "近1年", "全部"]
            # 預設選「近1季」
            selected_range = st.select_slider("顯示區間", options=range_options, value="近1季") 

        # --- 數據處理 ---
        df = hist_data.copy() # 用本地變數方便操作
        
        # 1. 移動平均線
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 2. 智慧波段識別 (Fractal) 為符合現實，增加判斷週期
        period = 5 
        # 使用 apply 或簡單的 loop 做更精確的「轉折」判斷
        # 這裡為了效能還是用 rolling，但邏輯不變
        df['Is_Support'] = df['Low'] == df['Low'].rolling(window=period*2+1, center=True).min()
        df['Is_Resistance'] = df['High'] == df['High'].rolling(window=period*2+1, center=True).max()
        
        # 3. 繪製「有效」的支撐壓力延伸線
        layout_shapes = []
        
        # 重設 index 以便操作
        df_reset = df.reset_index()
        dates = df_reset.iloc[:, 0]
        highs = df_reset['High'].values
        lows = df_reset['Low'].values
        closes = df_reset['Close'].values
        volumes = df_reset['Volume'].values # 引入成交量
        is_sup = df_reset['Is_Support'].values
        is_res = df_reset['Is_Resistance'].values
        
        n_candles = len(df_reset)
        
        # 保存有效的支撐/壓力線集合
        active_lines = [] 
        
        # 價位聚合容忍度 (例如 2%)
        tolerance = 0.02
        
        # 將所有轉折點收集起來
        pivots = []
        for i in range(n_candles - period):
            if is_sup[i]:
                pivots.append({'price': lows[i], 'index': i, 'type': 'support', 'volume': volumes[i]})
            if is_res[i]:
                pivots.append({'price': highs[i], 'index': i, 'type': 'resistance', 'volume': volumes[i]})
        
        # --- 趨勢線演算法 (Trend Lines) ---
        # 參考邏輯：連接兩個以上的轉折點，並確認中間沒有明顯突破
        
        trend_lines = []
        min_distance = 10 # 兩個點至少間隔 10 天，避免太短的線
        
        # 1. 上升/支撐趨勢線 (Support Line) - 連接低點
        sup_indices = [i for i, x in enumerate(is_sup) if x]
        # 為了效能與實用性，只看最近一年的資料點 (約 250 天)
        start_check = max(0, n_candles - 250)
        sup_indices = [i for i in sup_indices if i >= start_check]

        for i in range(len(sup_indices)):
            for j in range(i + 1, len(sup_indices)):
                idx1 = sup_indices[i]
                idx2 = sup_indices[j]
                
                if idx2 - idx1 < min_distance: continue
                
                x1, y1 = idx1, lows[idx1]
                x2, y2 = idx2, lows[idx2]
                
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # 驗證：中間的 K 線 (Close) 是否都在線之上 (容忍一點點誤差)
                # 這保證了這條線在過去是有支撐作用的
                valid = True
                for k in range(idx1 + 1, idx2):
                    expected_y = slope * k + intercept
                    if closes[k] < expected_y * 0.99: # 允許 1% 的假跌破
                        valid = False
                        break
                
                if valid:
                    # 延伸到最新日期
                    current_idx = n_candles - 1
                    projected_y = slope * current_idx + intercept
                    
                    # 檢查延伸段的有效性
                    # 如果這條線延伸到今天，價格卻已經遠遠跌破它，那它就失效了
                    # 我們只保留「價格還在線附近」或是「剛跌破但可能反壓」的線
                    
                    # 檢查最後一天的狀況
                    current_price = closes[current_idx]
                    
                    # 如果現價低於趨勢線超過 3%，視為已無效/已遠離
                    if current_price < projected_y * 0.97:
                         continue

                    # 3. 檢查「有效跌破」(Visual check only)
                    final_idx = n_candles - 1
                    is_terminated = False
                    
                    for k in range(idx2 + 5, n_candles):
                        proj_y = slope * k + intercept
                        close_k = closes[k]
                        
                        # 支撐線：若收盤價 明顯 < 趨勢線 (跌破)
                        if close_k < proj_y * 0.98: # 2% 跌破
                             final_idx = k
                             is_terminated = True
                             break
                    
                    final_y = slope * final_idx + intercept
                    mid_y = slope * idx2 + intercept
                         
                    # 計算得分：長度
                    score = (idx2 - idx1)
                    
                    trend_lines.append({
                        'x1': dates[idx1], 'y1': y1,
                        'x_mid': dates[idx2], 'y_mid': mid_y,
                        'x2': dates[final_idx], 'y2': final_y,
                        'type': 'support',
                        'score': score,
                        'terminated': is_terminated
                    })

        # 2. 下降/壓力趨勢線 (Resistance Line) - 連接高點
        res_indices = [i for i, x in enumerate(is_res) if x]
        res_indices = [i for i in res_indices if i >= start_check]

        for i in range(len(res_indices)):
            for j in range(i + 1, len(res_indices)):
                idx1 = res_indices[i]
                idx2 = res_indices[j]
                
                if idx2 - idx1 < min_distance: continue
                
                x1, y1 = idx1, highs[idx1]
                x2, y2 = idx2, highs[idx2]
                
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                valid = True
                for k in range(idx1 + 1, idx2):
                    expected_y = slope * k + intercept
                    if closes[k] > expected_y * 1.01: # 允許 1% 的假突破
                        valid = False
                        break
                
                if valid:
                    current_idx = n_candles - 1
                    projected_y = slope * current_idx + intercept
                    
                    current_price = closes[current_idx]
                    # 如果現價高於趨勢線超過 3%，視為已突破遠離
                    if current_price > projected_y * 1.03:
                        continue

                    # 3. 檢查「有效突破」(Visual check only)
                    # 我們將從 idx2 開始往後檢查，一旦被明顯突破就停止延伸，避免畫出已經無效的線
                    final_idx = n_candles - 1
                    is_terminated = False
                    
                    for k in range(idx2 + 5, n_candles): # 給一點緩衝，idx2 後 5 天再開始查
                        proj_y = slope * k + intercept
                        close_k = closes[k]
                        
                        # 壓力線：若收盤價 明顯 > 趨勢線 (突破)
                        if close_k > proj_y * 1.02: # 2% 突破
                             final_idx = k
                             is_terminated = True
                             break
                    
                    final_y = slope * final_idx + intercept
                    mid_y = slope * idx2 + intercept

                    score = (idx2 - idx1)
                    
                    trend_lines.append({
                        'x1': dates[idx1], 'y1': y1,
                        'x_mid': dates[idx2], 'y_mid': mid_y, # 中繼點
                        'x2': dates[final_idx], 'y2': final_y,
                        'type': 'resistance',
                        'score': score,
                        'terminated': is_terminated
                    })

        # 篩選並繪製
        # 每個類型只取最長的 1 條，確保畫面乾淨，避免多條線混淆
        trend_lines.sort(key=lambda x: x['score'], reverse=True)
        
        # 分組篩選
        supports = [l for l in trend_lines if l['type'] == 'support'][:1]
        resistances = [l for l in trend_lines if l['type'] == 'resistance'][:1]
        final_trend_lines = supports + resistances
        
        layout_annotations = [] # 初始化註解列表

        for line in final_trend_lines:
            color = '#26a69a' if line['type'] == 'support' else '#ef5350'
            
            # --- 視覺優化：分段顯示 ---
            # 1. 實線段 (Solid)：從 起點(x1) 到 確認點(mid/x2)
            #    這代表「形成趨勢」的基礎段
            
            # 如果這條線很短就結束了(被突破)，那可能就直接畫完
            # 我們這裡統一畫法：
            # 若 line['terminated'] 為 True，則畫到 termination point
            # 若 False，則延伸到最後
            
            # 第一段：實線 (基礎)
            layout_shapes.append(dict(
                type="line",
                x0=line['x1'], y0=line['y1'],
                x1=line['x_mid'], y1=line['y_mid'],
                line=dict(color=color, width=2), # 實線
                layer="below"
            ))
            
            # 第二段：虛線 (延伸/預測)
            if line['x_mid'] != line['x2']:
                layout_shapes.append(dict(
                    type="line",
                    x0=line['x_mid'], y0=line['y_mid'],
                    x1=line['x2'], y1=line['y2'],
                    line=dict(color=color, width=2, dash="dash"), # 虛線
                    layer="below"
                ))
            
            # 標示目前的延伸價格 (及狀態)
            label_text = f"{line['y2']:.2f}"
            if line.get('terminated'): label_text += " (已突破)"
                
            layout_annotations.append(dict(
                x=line['x2'], y=line['y2'],
                text=label_text,
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color=color)
            ))
        
        # --- 由於 subplot 移除，我們需要重新建立單一圖表的 Figure 物件 ---
        # 為了保留上方的邏輯 (layout_shapes)，我們在這裡重新組裝
        
        # 重新建立 Figure (單一圖表)
        fig = go.Figure()

        # 將收集到的 annotation 加入
        for ann in layout_annotations:
            fig.add_annotation(ann)

        
        # 1. K線
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        increasing_line_color='#ef5350', 
                        decreasing_line_color='#26a69a', 
                        name="K線"))
        
        # 2. 均線
        # 將均線與轉折點設為 xaxis='x2' (副 X 軸)，目的是讓它們不要出現在底部的 Range Slider 中
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ffa726', width=1.5), name="MA20", xaxis="x2"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#42a5f5', width=1.5), name="MA60", xaxis="x2"))
        
        # 3. 轉折點
        sup_points = df[df['Is_Support']]
        res_points = df[df['Is_Resistance']]
        fig.add_trace(go.Scatter(x=sup_points.index, y=sup_points['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=8, color='#26a69a'), name='支撐轉折', xaxis="x2"))
        fig.add_trace(go.Scatter(x=res_points.index, y=res_points['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=8, color='#ef5350'), name='壓力轉折', xaxis="x2"))

        # 應用 Layout
        fig.update_layout(shapes=layout_shapes)

        fig.update_layout(
            height=600,
            dragmode="pan", # 設定滑鼠預設行為為平移 (Pan)，而非框選縮放
            xaxis_title="日期",
            yaxis_title="價格",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            # 打開 Range Slider (底部拉取範圍)
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.15), 
                type="date"
            )
        )
        
        # 根據選擇調整 Zoom
        lookback_map = {
            "近1週": 5,
            "近1月": 22,
            "近1季": 66,
            "近半年": 132,
            "近1年": 252
        }
        
        # --- 自動計算 Y 軸範圍 (Auto Scale Y-Axis) ---
        # 為了避免 Trend Lines (趨勢線) 延伸到很遠的地方導致 Y軸 被拉到看不到 K 線
        # 我們必須強制根據「可視範圍內的 K 線價格」來設定 Y 軸
        
        # 預設：使用全部資料
        visible_df = df
        
        if selected_range in lookback_map:
            days = lookback_map[selected_range]
            if len(df) > days:
                start_date = df.index[-days]
                end_date = df.index[-1]
                fig.update_xaxes(range=[start_date, end_date])
                visible_df = df.iloc[-days:]

        # 計算價格範圍
        if not visible_df.empty:
            min_price = visible_df['Low'].min()
            max_price = visible_df['High'].max()
            
            # 加上 5% 緩衝
            buffer = (max_price - min_price) * 0.05
            if buffer == 0: buffer = max_price * 0.01
            yaxis_range = [min_price - buffer, max_price + buffer]
        else:
            yaxis_range = None

        # 應用 Layout
        fig.update_layout(shapes=layout_shapes)

        fig.update_layout(
            height=600,
            xaxis_title="日期",
            yaxis_title="價格",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            # 打開 Range Slider (底部拉取範圍)
            xaxis=dict(
                rangeslider=dict(
                    visible=True, 
                    thickness=0.15
                ), 
                type="date"
            ),
            # 建立副 X 軸與主軸同步，但不顯示 Range Slider
            xaxis2=dict(
                overlaying="x",
                matches="x", 
                visible=False,
                type="date"
            ),
            yaxis=dict(
                range=yaxis_range, # 強制設定範圍
                autorange=False,   # 關閉自動範圍以免被 shape 影響
                fixedrange=False   # 允許使用者手動縮放
            )
        )
        
        # 修正：強制 Range Slider 的 Y 軸自動縮放，避免被主圖的 Y 軸鎖定導致顯示異常
        fig.update_layout(xaxis_rangeslider_yaxis_rangemode="auto")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 訊號說明：\n- **K 線圖**：包含均線 (MA20/60) 與支撐壓力位。\n- **操作**：滑鼠左鍵拖曳可平移圖表，滾輪可縮放時間軸。")
    else:
        st.warning(f"找不到代碼 {symbol} 的數據，請檢查是否輸入正確。")
