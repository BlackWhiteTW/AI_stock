import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon

def render_detailed_statements(ticker, financial_mgr, force_update=False):
    """詳細財務報表子模組 (損益、資產負債、現金流)"""
    header_with_icon("詳細財務報表", "📋", "查看季度的原始數據與變動")
    
    with st.expander("🔍 點擊展開詳細報表數據", expanded=True):
        financial_data = financial_mgr.get_financial_statements(ticker, force_update=force_update)
        
        if not financial_data:
            st.warning("⚠️ 暫無詳細報表數據")
            return

        stmt_tab1, stmt_tab2, stmt_tab3 = st.tabs(["損益表", "資產負債表", "現金流量表"])
        
        with stmt_tab1:
            income_stmt = financial_data.get('income_statement', {})
            if income_stmt:
                st.dataframe(pd.DataFrame(income_stmt), use_container_width=True)
            else:
                st.info("無季度損益表數據")
        
        with stmt_tab2:
            balance_sheet = financial_data.get('balance_sheet', {})
            if balance_sheet:
                st.dataframe(pd.DataFrame(balance_sheet), use_container_width=True)
            else:
                st.info("無季度資產負債表數據")
                
        with stmt_tab3:
            cash_flow = financial_data.get('cash_flow', {})
            if cash_flow:
                st.dataframe(pd.DataFrame(cash_flow), use_container_width=True)
            else:
                st.info("無季度現金流量表數據")
