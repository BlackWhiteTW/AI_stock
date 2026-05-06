import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon, status_card


def render_financial_ratios(summary):
    """財務比率與指標分析子模組"""
    header_with_icon("核心財務指標", "", "衡量公司的獲利與償債能力")

    if summary.get("status") == "etf":
        st.info(summary.get("message", "ETF 不適用傳統財報指標，以下顯示持股概況。"))

        profile = summary.get("etf_profile", {})
        stock_info = profile.get("stock_info", {}) or {}

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

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "持股檔數",
            (
                profile.get("holding_count")
                if profile.get("holding_count") is not None
                else "未取得"
            ),
        )
        c2.metric("資料來源", profile.get("source", "Yahoo TW"))
        c3.metric("最近更新", str(profile.get("update_time", "N/A"))[:10])

        st.subheader("ETF 概況")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric(
            "市價",
            f"{profile.get('market_price', stock_info.get('regularMarketPrice', 'N/A'))}",
        )
        o2.metric(
            "淨值", f"{profile.get('nav_price', stock_info.get('navPrice', 'N/A'))}"
        )
        o3.metric(
            "殖利率",
            _fmt_percent(
                profile.get("dividend_yield", stock_info.get("dividendYield"))
            ),
        )
        o4.metric(
            "費用率",
            _fmt_percent(
                profile.get("expense_ratio", stock_info.get("netExpenseRatio"))
            ),
        )

        o5, o6, o7, o8 = st.columns(4)
        total_assets = profile.get(
            "total_assets", stock_info.get("netAssets") or stock_info.get("totalAssets")
        )
        try:
            total_assets_text = (
                f"{float(total_assets) / 100000000:,.0f} 億" if total_assets else "N/A"
            )
        except Exception:
            total_assets_text = str(total_assets) if total_assets else "N/A"
        o5.metric("淨資產", total_assets_text)
        o6.metric(
            "今年以來報酬",
            _fmt_return(profile.get("ytd_return", stock_info.get("ytdReturn"))),
        )
        o7.metric(
            "近三年平均報酬",
            _fmt_return(
                profile.get(
                    "three_year_return", stock_info.get("threeYearAverageReturn")
                )
            ),
        )
        o8.metric(
            "近五年平均報酬",
            _fmt_return(
                profile.get("five_year_return", stock_info.get("fiveYearAverageReturn"))
            ),
        )

        beta_value = profile.get(
            "beta", stock_info.get("beta3Year") or stock_info.get("beta")
        )
        pe_value = profile.get("pe_ratio", stock_info.get("trailingPE"))
        st.caption(
            f"基金名稱: {profile.get('fund_name') or stock_info.get('longName') or stock_info.get('shortName') or 'N/A'}"
        )
        st.caption(
            f"本益比: {pe_value if pe_value is not None else 'N/A'} | 三年 Beta: {beta_value if beta_value is not None else 'N/A'}"
        )

        top_holdings = profile.get("top_holdings", [])
        if top_holdings:
            st.caption("前十大持股")
            st.dataframe(
                pd.DataFrame(top_holdings), use_container_width=True, hide_index=True
            )
        else:
            st.info("目前未取得持股明細，先顯示資料庫中的 ETF 概況。")

        sector_weights = profile.get("sector_weights", [])
        if sector_weights:
            st.caption("產業配置")
            st.dataframe(
                pd.DataFrame(sector_weights), use_container_width=True, hide_index=True
            )

        raw_summary = profile.get("raw_summary", {})
        if raw_summary:
            st.caption("持股摘要")
            st.json(raw_summary)
        elif stock_info:
            st.caption("資料庫已有資料")
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
            ]
            st.dataframe(
                pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True
            )

        return

    # 財務健康評分顯示
    score = summary.get("health_score", 50)
    rating = summary.get("rating", "B 中等")
    status_card(
        "財務健康度總分",
        f"{score} 分",
        delta=rating,
        help_text="綜合考量多維度財務數據",
    )

    tab_profit, tab_solvency, tab_valuation = st.tabs(
        [" 獲利能力", "️ 償債能力", " 估值指標"]
    )

    with tab_profit:
        _render_profitability(summary.get("profitability", {}))

    with tab_solvency:
        _render_solvency(summary.get("solvency", {}))

    with tab_valuation:
        _render_valuation(summary.get("valuation", {}))


def _render_profitability(profitability):
    """獲利指標細節"""
    c1, c2, c3 = st.columns(3)
    c1.metric("ROE (股東權益報酬率)", f"{profitability.get('roe', 0):.2f}%")
    c2.metric("ROA (資產報酬率)", f"{profitability.get('roa', 0):.2f}%")
    c3.metric("淨利率", f"{profitability.get('net_profit_margin', 0):.2f}%")


def _render_solvency(solvency):
    """償債指標細節"""
    c1, c2 = st.columns(2)
    c1.metric("負債比率", f"{solvency.get('debt_ratio', 0):.2f}%")
    c2.metric("流動比率", f"{solvency.get('current_ratio', 0):.2f}")


def _render_valuation(valuation):
    """估值指標細節"""
    c1, c2, c3 = st.columns(3)
    c1.metric("P/E (本益比)", f"{valuation.get('pe_ratio', 0):.2f}")
    c2.metric("P/B (股價淨值比)", f"{valuation.get('pb_ratio', 0):.2f}")
    mv = valuation.get("market_cap", 0) / 1e8
    c3.metric("總市值 (億)", f"${mv:,.0f}")
