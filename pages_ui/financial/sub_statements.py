import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon


def render_detailed_statements(
    ticker, financial_mgr, force_update=False, is_etf=False, summary=None
):
    """詳細財務報表子模組 (根據類型顯示不同內容)"""
    header_with_icon("詳細財務報表", "", "查看季度的原始數據與變動")

    if is_etf and summary:
        # ETF 專用顯示
        _render_etf_details(summary)
    else:
        # 一般股票的傳統財務報表
        _render_stock_statements(ticker, financial_mgr, force_update)


def _render_etf_details(summary):
    """ETF 詳細資訊顯示"""
    etf_profile = summary.get("etf_profile", {})
    stock_info = etf_profile.get("stock_info", {}) or {}

    def _fmt_percent(value):
        if value is None or value == "":
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return str(value)

    def _fmt_return(value):
        if value is None or value == "":
            return "N/A"
        try:
            numeric_value = float(value)
            if abs(numeric_value) <= 1.5:
                numeric_value *= 100
            return f"{numeric_value:.2f}%"
        except Exception:
            return str(value)

    with st.expander("點擊展開詳細報表數據", expanded=True):
        st.info("ETF 不適用傳統損益表/資產負債表分析，以下為 ETF 特有的持股資訊。")

        # 1. 基本資訊
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "持股檔數",
            (
                etf_profile.get("holding_count")
                if etf_profile.get("holding_count") is not None
                else "未取得"
            ),
        )
        col2.metric("資料來源", etf_profile.get("source", "Yahoo TW"))
        col3.metric("最近更新", str(etf_profile.get("update_time", "N/A"))[:10])

        st.subheader("ETF 概況")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "市價",
            stock_info.get(
                "regularMarketPrice", etf_profile.get("market_price", "N/A")
            ),
        )
        s2.metric(
            "淨值", stock_info.get("navPrice", etf_profile.get("nav_price", "N/A"))
        )
        s3.metric(
            "殖利率",
            _fmt_percent(
                etf_profile.get("dividend_yield", stock_info.get("dividendYield"))
            ),
        )
        s4.metric(
            "費用率",
            _fmt_percent(
                etf_profile.get("expense_ratio", stock_info.get("netExpenseRatio"))
            ),
        )

        s5, s6, s7, s8 = st.columns(4)
        total_assets = etf_profile.get(
            "total_assets", stock_info.get("netAssets") or stock_info.get("totalAssets")
        )
        try:
            total_assets_text = (
                f"{float(total_assets) / 100000000:,.0f} 億" if total_assets else "N/A"
            )
        except Exception:
            total_assets_text = str(total_assets) if total_assets else "N/A"
        s5.metric("淨資產", total_assets_text)
        s6.metric(
            "今年以來報酬",
            _fmt_return(etf_profile.get("ytd_return", stock_info.get("ytdReturn"))),
        )
        s7.metric(
            "近三年平均報酬",
            _fmt_return(
                etf_profile.get(
                    "three_year_return", stock_info.get("threeYearAverageReturn")
                )
            ),
        )
        s8.metric(
            "近五年平均報酬",
            _fmt_return(
                etf_profile.get(
                    "five_year_return", stock_info.get("fiveYearAverageReturn")
                )
            ),
        )

        st.caption(
            f"基金名稱: {etf_profile.get('fund_name') or stock_info.get('longName') or stock_info.get('shortName') or 'N/A'}"
        )
        st.caption(
            f"本益比: {etf_profile.get('pe_ratio', stock_info.get('trailingPE')) or 'N/A'} | 三年 Beta: {etf_profile.get('beta', stock_info.get('beta3Year') or stock_info.get('beta')) or 'N/A'}"
        )

        st.divider()

        # 2. 前十大持股
        top_holdings = etf_profile.get("top_holdings", [])
        if top_holdings:
            st.subheader("前十大持股")
            holdings_data = []
            for i, holding in enumerate(top_holdings[:10], 1):
                if isinstance(holding, dict):
                    holdings_data.append(
                        {
                            "排名": i,
                            "代碼": holding.get("code", "N/A"),
                            "公司": holding.get("name", "N/A"),
                            "占比": f"{holding.get('percentage', 0):.2f}%",
                        }
                    )
            if holdings_data:
                st.dataframe(
                    pd.DataFrame(holdings_data),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("目前沒有持股明細，改顯示資料庫中已存在的 ETF 概況。")

        st.divider()

        # 3. 產業配置
        sector_weights = etf_profile.get("sector_weights", [])
        if sector_weights:
            st.subheader("產業配置")
            sector_data = []
            for sector in sector_weights:
                if isinstance(sector, dict):
                    sector_data.append(
                        {
                            "產業": sector.get("sector", "N/A"),
                            "占比": f"{sector.get('percentage', 0):.2f}%",
                        }
                    )
            if sector_data:
                st.dataframe(
                    pd.DataFrame(sector_data), use_container_width=True, hide_index=True
                )

                # 餅圖展示
                sector_names = [s.get("sector", "N/A") for s in sector_weights]
                sector_percentages = [
                    float(s.get("percentage", 0)) for s in sector_weights
                ]

                if sector_names and any(sector_percentages):
                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure(
                            data=[
                                go.Pie(labels=sector_names, values=sector_percentages)
                            ]
                        )
                        fig.update_layout(title="產業配置分布", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
        elif stock_info:
            st.subheader("資料庫已有資料")
            fallback_rows = [
                {
                    "項目": "基金名稱",
                    "值": stock_info.get("longName") or stock_info.get("shortName"),
                },
                {"項目": "市價", "值": stock_info.get("regularMarketPrice")},
                {"項目": "淨值", "值": stock_info.get("navPrice")},
                {"項目": "殖利率", "值": stock_info.get("dividendYield")},
                {"項目": "費用率", "值": stock_info.get("netExpenseRatio")},
                {
                    "項目": "淨資產",
                    "值": stock_info.get("netAssets") or stock_info.get("totalAssets"),
                },
                {"項目": "今年以來報酬", "值": stock_info.get("ytdReturn")},
                {
                    "項目": "近三年平均報酬",
                    "值": stock_info.get("threeYearAverageReturn"),
                },
                {
                    "項目": "近五年平均報酬",
                    "值": stock_info.get("fiveYearAverageReturn"),
                },
            ]
            st.dataframe(
                pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True
            )

        st.divider()

        # 4. 持股摘要
        raw_summary = etf_profile.get("raw_summary", {})
        if raw_summary:
            st.subheader("持股摘要")
            summary_items = []
            for key, value in raw_summary.items():
                if value:
                    summary_items.append(f"**{key}**: {value}")
            if summary_items:
                for item in summary_items:
                    st.write(item)
        elif stock_info:
            st.caption("資料庫原始資料")
            st.json(
                {
                    "基金名稱": stock_info.get("longName")
                    or stock_info.get("shortName"),
                    "市價": stock_info.get("regularMarketPrice"),
                    "淨值": stock_info.get("navPrice"),
                    "殖利率": stock_info.get("dividendYield"),
                    "費用率": stock_info.get("netExpenseRatio"),
                    "淨資產": stock_info.get("netAssets")
                    or stock_info.get("totalAssets"),
                    "今年以來報酬": stock_info.get("ytdReturn"),
                }
            )

        st.divider()

        # 5. 損益資訊 (如有)
        eps = etf_profile.get("eps", {})
        revenue = etf_profile.get("revenue", {})

        if eps or revenue:
            st.subheader("財務績效")
            col1, col2 = st.columns(2)

            if eps:
                with col1:
                    st.write("**每股盈餘 (EPS)**")
                    for period, value in list(eps.items())[:5]:
                        st.write(f"{period}: {value}")

            if revenue:
                with col2:
                    st.write("**營收**")
                    for period, value in list(revenue.items())[:5]:
                        st.write(f"{period}: {value}")


def _render_stock_statements(ticker, financial_mgr, force_update=False):
    """一般股票的傳統財務報表顯示"""
    with st.expander("點擊展開詳細報表數據", expanded=True):
        financial_data = financial_mgr.get_financial_statements(
            ticker, force_update=force_update
        )

        if not financial_data:
            st.warning("暫無詳細報表數據")
            return

        stmt_tab1, stmt_tab2, stmt_tab3 = st.tabs(
            ["損益表", "資產負債表", "現金流量表"]
        )

        with stmt_tab1:
            income_stmt = financial_data.get("income_statement", {})
            if income_stmt:
                st.dataframe(pd.DataFrame(income_stmt), use_container_width=True)
            else:
                st.info("無季度損益表數據")

        with stmt_tab2:
            balance_sheet = financial_data.get("balance_sheet", {})
            if balance_sheet:
                st.dataframe(pd.DataFrame(balance_sheet), use_container_width=True)
            else:
                st.info("無季度資產負債表數據")

        with stmt_tab3:
            cash_flow = financial_data.get("cash_flow", {})
            if cash_flow:
                st.dataframe(pd.DataFrame(cash_flow), use_container_width=True)
            else:
                st.info("無季度現金流量表數據")
