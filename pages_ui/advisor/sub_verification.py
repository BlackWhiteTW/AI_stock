import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from datetime import datetime


def render_prediction_analysis(ai_advisor, df_stocks):
    """
    🕵️‍♀️ 每日 AI 預測驗證與分析中心
    """
    st.header("🕵️‍♀️ 每日 AI 預測分析")

    # 頂部控制列 (Control Panel)
    st.subheader("⚙️ 手動執行與控制")
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 1, 1, 1])

    is_weekend = datetime.now().weekday() >= 5

    with col_ctrl1:
        if is_weekend:
            st.warning("⚠️ 今日為週末，股市未開盤。自動預測已暫停以避免參數失準。")
        else:
            st.info("💡 系統每日於 08:30 自動預測、14:30 自動驗證。")

    with col_ctrl2:
        btn_label = "🚀 執行預測" if not is_weekend else "⚠️ 週末強制預測"
        if st.button(
            btn_label,
            use_container_width=True,
            help="立即執行開盤前 AI 預測任務",
            type="secondary" if is_weekend else "primary",
        ):
            with st.spinner("正在進行預測..."):
                # 週末執行時強制 force=True
                count = ai_advisor.run_morning_predictions(force=is_weekend)
                if count > 0:
                    st.success(f"已完成 {count} 檔模型預測！")
                    st.rerun()
                else:
                    st.info("今日無預測任務執行。")
    with col_ctrl3:
        force_val = st.checkbox(
            "🚩 強制驗證所有",
            help="忽略日期檢查，將最新價格填入所有等待中的預測",
            value=False,
        )
        if st.button(
            "🔍 執行驗證", use_container_width=True, help="獲取目前收盤價並驗證"
        ):
            with st.spinner("正在驗證..."):
                count, _ = ai_advisor.validate_afternoon_results(force=force_val)
                if count > 0:
                    st.success(f"✅ 已完成 {count} 筆數據驗證！")
                    st.rerun()
                else:
                    st.warning("⚠️ 未發現可驗證數據。")
    with col_ctrl4:
        if st.button("🔄 重新整理", use_container_width=True):
            st.rerun()

    # 讀取數據
    if not os.path.exists(ai_advisor.prediction_file):
        st.warning("⚠️ 尚無預測紀錄檔案。請點擊上方「🚀 執行預測」開始。")
        return

    try:
        with open(ai_advisor.prediction_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            st.info("📭 紀錄檔案為空。")
            return

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        # --- 數據統計區 ---
        st.divider()
        st.subheader("📊 績效概覽")

        # 只計算已驗證的 (有實際價格的)
        df_valid = df.dropna(subset=["actual_price"]).copy()

        if not df_valid.empty:
            # 計算準確率百分比
            df_valid["error_pct"] = (
                abs(df_valid["predicted_price"] - df_valid["actual_price"])
                / df_valid["actual_price"]
                * 100
            )
            avg_error = df_valid["error_pct"].mean()

            # 計算勝率 (方向預測正確率)
            # 方向正確: (預測價 > 基準價 且 實際價 > 基準價) 或 (預測價 < 基準價 且 實際價 < 基準價)
            df_valid["dir_correct"] = (
                (df_valid["predicted_price"] > df_valid["base_price"])
                & (df_valid["actual_price"] > df_valid["base_price"])
            ) | (
                (df_valid["predicted_price"] < df_valid["base_price"])
                & (df_valid["actual_price"] < df_valid["base_price"])
            )
            hit_rate = df_valid["dir_correct"].mean() * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總預測數", f"{len(df)} 筆")
            m2.metric("已驗證數", f"{len(df_valid)} 筆")
            m3.metric(
                "平均誤差",
                f"{avg_error:.2f}%",
                delta=f"{-avg_error:.2f}%",
                delta_color="normal",
            )
            m4.metric("趨勢勝率", f"{hit_rate:.1f}%")
        else:
            st.warning(
                "⏳ 今日預測尚未驗證 (收盤後 14:30 將自動執行，或點擊上方「🔍 執行驗證」)。"
            )

        # --- 詳細資料表與過濾 ---
        st.divider()
        st.subheader("📝 詳細預測紀錄")

        # 篩選器
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            all_dates = sorted(df["date"].dt.date.unique().tolist(), reverse=True)
            date_options = ["全部日期"] + [d.strftime("%Y-%m-%d") for d in all_dates]
            selected_date = st.selectbox("📅 選擇日期", date_options)

        with col_f2:
            all_tickers = sorted(df["ticker"].unique().tolist())
            ticker_options = ["全部標的"] + all_tickers
            selected_ticker = st.selectbox("🔍 搜尋標的", ticker_options)

        # 執行篩選
        df_display = df.copy()
        if selected_date != "全部日期":
            df_display = df_display[
                df_display["date"].dt.date
                == datetime.strptime(selected_date, "%Y-%m-%d").date()
            ]
        if selected_ticker != "全部標的":
            df_display = df_display[df_display["ticker"] == selected_ticker]

        # 顯示表格
        st.dataframe(
            df_display.sort_values(by="date", ascending=False),
            column_config={
                "date": "預測日期",
                "ticker": "代碼",
                "base_price": st.column_config.NumberColumn("昨收價", format="%.2f"),
                "predicted_price": st.column_config.NumberColumn(
                    "預測價", format="%.2f"
                ),
                "expected_change_pct": st.column_config.NumberColumn(
                    "預期漲跌", format="%.2f%%"
                ),
                "actual_price": st.column_config.NumberColumn("實際價", format="%.2f"),
                "diff_pct": st.column_config.NumberColumn("預測誤差", format="%.2f%%"),
                "accuracy": "準確度",
            },
            use_container_width=True,
            hide_index=True,
        )

        # --- 趨勢圖表 ---
        if selected_ticker != "全部標的":
            st.subheader(f"📈 {selected_ticker} 預測趨勢追蹤")
            ticker_df = df[df["ticker"] == selected_ticker].sort_values("date")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["actual_price"],
                    name="實際價",
                    line=dict(color="white", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["predicted_price"],
                    name="預測價",
                    line=dict(color="#00FF00", dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["base_price"],
                    name="昨收參考",
                    line=dict(color="gray", width=1),
                )
            )

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"分析預測數據時發生錯誤: {e}")
        import traceback

        st.code(traceback.format_exc())
