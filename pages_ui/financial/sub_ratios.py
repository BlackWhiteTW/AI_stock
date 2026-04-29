import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon, status_card


def render_financial_ratios(summary):
    """財務比率與指標分析子模組"""
    header_with_icon("核心財務指標", "📊", "衡量公司的獲利與償債能力")

    if summary.get("status") == "etf":
        st.info(summary.get("message", "ETF 不適用傳統財報指標，以下顯示持股概況。"))

        profile = summary.get("etf_profile", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("持股檔數", profile.get("holding_count", 0))
        c2.metric("資料來源", profile.get("source", "Yahoo TW"))
        c3.metric("最近更新", profile.get("update_time", "N/A"))

        top_holdings = profile.get("top_holdings", [])
        if top_holdings:
            st.caption("前十大持股")
            st.dataframe(
                pd.DataFrame(top_holdings), use_container_width=True, hide_index=True
            )

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
        ["💹 獲利能力", "🛡️ 償債能力", "💎 估值指標"]
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
