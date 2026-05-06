import streamlit as st
from services.llm_service import LLMService
from services.search_optimizer import SearchOptimizer

# 初始化服務
llm_service = LLMService()
search_optimizer = SearchOptimizer()


def render_ai_chat(symbol, current_price, ai_advisor):
    """AI 分析師聊天室子模組"""
    st.subheader(f"正在與 {symbol} 專屬分析師對話")
    st.info(
        "您可以詢問關於這檔股票的任何問題，AI 會結合目前的技術面、基本面與即時新聞為您解答。"
    )

    # 初始化聊天紀錄
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 顯示聊天紀錄
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 快速提問按鈕
    quick_queries = [
        f"總結 {symbol} 的當前多空力道",
        f"根據最新新聞，{symbol} 有什麼潛在利多？",
        f"如果我現在買入 {symbol}，停損點建議設在哪？",
        f"這檔股票適合長期持有嗎？",
    ]

    cols = st.columns(2)
    for i, q in enumerate(quick_queries):
        if cols[i % 2].button(q, key=f"q_{i}", use_container_width=True):
            st.session_state.temp_query = q

    # 聊天輸入框
    if prompt := st.chat_input("輸入您的問題..."):
        st.session_state.temp_query = prompt

    # 處理查詢
    if "temp_query" in st.session_state:
        query = st.session_state.temp_query
        del st.session_state.temp_query

        # 1. 顯示使用者問題
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. 獲取背景資訊
        with st.spinner("AI 分析師正在閱讀報告與即時新聞..."):
            try:
                summary = search_optimizer.get_cached_summary(symbol, ai_advisor)

                # 構建 Context
                context = f"""
                用戶當前關注的股票: {symbol}
                當前價格: {current_price}
                技術趨勢: {summary['technical']['trend']['daily']}
                最新新聞摘要: {summary['web_news']}
                AI 原始診斷: {summary['ai_verdict']}
                """

                full_prompt = f"背景資訊如下：\n{context}\n\n請根據以上資訊，以專業分析師的口吻回答用戶的問題：{query}"
                response = llm_service.call_gemini(full_prompt)

                # 3. 顯示 AI 回覆
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"分析師暫時忙碌中: {e}")
