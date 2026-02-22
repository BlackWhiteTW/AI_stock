import streamlit as st
import pandas as pd

def render_tab_history(account):
    st.subheader("歷史交易紀錄")
    if account.history:
        df_history = pd.DataFrame(account.history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("暫無交易紀錄")
