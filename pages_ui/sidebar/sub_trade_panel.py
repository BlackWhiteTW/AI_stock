import streamlit as st

def render_trade_panel(account, symbol, current_price):
    """側邊欄下單操作子模組"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("下單")
    
    action = st.sidebar.radio("動作", ["買入", "賣出"], horizontal=True)
    qty = st.sidebar.number_input("數量 (股)", min_value=1, value=1000, step=100)
    
    if st.sidebar.button("送出訂單", use_container_width=True):
        if current_price > 0:
            if action == "買入":
                success, msg = account.buy(symbol, current_price, qty)
                if success: st.sidebar.success(msg)
                else: st.sidebar.error(msg)
            else:
                success, msg = account.sell(symbol, current_price, qty)
                if success: st.sidebar.success(msg)
                else: st.sidebar.error(msg)
        else:
            st.sidebar.error("無法取得現價")
