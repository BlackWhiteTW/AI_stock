"""Google Colab entry point for the project's PyTorch training services."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Stock training on Colab GPU")
    parser.add_argument(
        "--mode",
        choices=("base", "stock", "gnn"),
        default="base",
        help="base=market base model, stock=one or more stocks, gnn=market ST-GNN",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols, for example 2330.TW 0050.TW (required for stock/gnn)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="GNN epochs; individual LSTM training uses the service default",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use the smaller base-model configuration instead of enhanced mode",
    )
    return parser.parse_args()


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker.endswith((".TW", ".TWO")):
        ticker += ".TW"
    return ticker


def show_runtime() -> None:
    import torch

    print(f"Project root: {PROJECT_ROOT}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")


def run_base(enhanced: bool) -> None:
    from services.ai_service import AIService

    service = AIService()
    result = service.train_base_model(enhanced=enhanced)
    print(result)


def run_stock(tickers: list[str]) -> None:
    from services.ai_service import AIService
    from utils import get_stock_data

    service = AIService()
    for ticker in tickers:
        _, history, _, _ = get_stock_data(ticker)
        if history.empty:
            print(f"[SKIP] {ticker}: no market data")
            continue
        result, loss = service.train_deep_learning_model(ticker, history)
        print(f"{result} | loss={loss:.6f}")


def run_gnn(tickers: list[str], epochs: int) -> None:
    from services.gnn_service import GNNService

    service = GNNService()
    print(service.train_market_gnn(tickers=tickers, epochs=epochs))


def main() -> None:
    args = parse_args()
    show_runtime()

    if args.mode == "base":
        run_base(enhanced=not args.basic)
        return

    if not args.tickers:
        raise SystemExit("--tickers is required when --mode is stock or gnn")

    tickers = [normalize_ticker(ticker) for ticker in args.tickers]
    if args.mode == "stock":
        run_stock(tickers)
    else:
        run_gnn(tickers, args.epochs)


if __name__ == "__main__":
    main()