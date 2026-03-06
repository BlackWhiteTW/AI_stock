import streamlit as st
import pandas as pd

def header_with_icon(title, icon, subtitle=None):
    """帶有圖示的標準標題組件"""
    st.header(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)

def status_card(label, value, delta=None, help_text=None, is_success=True):
    """自定義統計卡片"""
    with st.container():
        st.metric(label=label, value=value, delta=delta, help=help_text)
        if is_success:
            st.markdown("---")

def info_box(title, items):
    """帶有條列式資訊的提示框"""
    info_text = f"**{title}**\n"
    for item in items:
        info_text += f"- {item}\n"
    st.info(info_text)

def action_button(label, key, type="primary", use_container_width=True):
    """標準化動作按鈕"""
    return st.button(label, key=key, type=type, use_container_width=use_container_width)

def data_table(df, height=400):
    """優化版數據表格顯示"""
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height
    )
