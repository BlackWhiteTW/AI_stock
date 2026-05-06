#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 ETF 和股票的差异化显示功能"""

from utils import get_tw_stock_list
from services.financial_data import get_financial_manager


def test_etf_classification():
    """测试 ETF 分类功能"""
    print("=" * 60)
    print("测试 ETF 和股票分类")
    print("=" * 60)

    financial_mgr = get_financial_manager()

    # 测试 0050 (ETF)
    print("\n1. 测试 0050 (元大台灣50 ETF):")
    summary_etf = financial_mgr.get_financial_summary("0050.TW")
    print(f"   状态: {summary_etf.get('status')}")
    print(f"   是否为 ETF: {summary_etf.get('status') == 'etf'}")
    if summary_etf.get("status") == "etf":
        etf_profile = summary_etf.get("etf_profile", {})
        print(f"   持股档数: {etf_profile.get('holding_count', 0)}")
        print(f"   顶部持股数: {len(etf_profile.get('top_holdings', []))}")
        print(f"   产业配置数: {len(etf_profile.get('sector_weights', []))}")

    # 测试 2330 (股票)
    print("\n2. 测试 2330 (台积电):")
    summary_stock = financial_mgr.get_financial_summary("2330.TW")
    print(f"   状态: {summary_stock.get('status')}")
    print(f"   是否为 ETF: {summary_stock.get('status') == 'etf'}")
    if summary_stock.get("status") != "etf":
        print(f"   财务健康度: {summary_stock.get('health_score', 0)}")
        print(f"   评级: {summary_stock.get('rating', 'N/A')}")

    print("\n✅ 分类测试完成！")


if __name__ == "__main__":
    test_etf_classification()
