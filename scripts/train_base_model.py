"""
訓練遷移學習基礎模型
使用台股大型權值股訓練通用模型，供後續個股快速訓練使用
"""
from services.ai_service import AIService

def main():
    # 選擇台股大型權值股（市值大、交易量大、具代表性）
    base_stocks = [
        '2330.TW',  # 台積電
        '2317.TW',  # 鴻海
        '2454.TW',  # 聯發科
        '2412.TW',  # 中華電
        '2882.TW',  # 國泰金
        '2308.TW',  # 台達電
        '2886.TW',  # 兆豐金
        '2891.TW',  # 中信金
    ]
    
    print("=" * 60)
    print("🚀 遷移學習基礎模型訓練程序")
    print("=" * 60)
    print(f"\n使用 {len(base_stocks)} 支大型權值股訓練通用模型")
    print("訓練完成後，所有個股訓練將自動加速 50-70%\n")
    
    service = AIService()
    
    def progress_callback(current, total, loss, elapsed):
        """訓練進度回調"""
        progress = (current / total) * 100
        print(f"  進度: {current}/{total} ({progress:.1f}%) | Loss: {loss:.6f} | 耗時: {elapsed:.1f}s")
    
    # 開始訓練基礎模型
    result = service.train_base_model(base_stocks, progress_callback=progress_callback)
    
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)

if __name__ == "__main__":
    main()
