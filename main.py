"""Daily orchestrator: fetch → analyze → score → report → deliver."""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime

from tqdm import tqdm

from config import RUN_MODE, TOP_N, WATCHLIST
from data_sources.yahoo import fetch_many
from data_sources.universe import broad_universe
from analysis.composite import analyze
from report_generator import write_reports, telegram_summary
from telegram_bot import send_message, send_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("daily")


def run(mode: str = RUN_MODE, top_n: int = TOP_N, send_tg: bool = True) -> None:
    log.info("=== Daily Stock Analysis @ %s | mode=%s ===", datetime.now().isoformat(timespec="seconds"), mode)

    if mode == "broad":
        symbols = broad_universe()
    else:
        symbols = WATCHLIST
    log.info("Universe: %d tickers", len(symbols))

    log.info("Fetching market data ...")
    data = fetch_many(symbols, period="1y")

    log.info("Analyzing ...")
    reports = []
    for sym, td in tqdm(data.items(), desc="Analyzing"):
        try:
            reports.append(analyze(td))
        except Exception as e:
            log.warning("analyze failed %s: %s", sym, e)

    reports = [r for r in reports if r.composite_score > 0]
    log.info("Analysis complete: %d valid reports", len(reports))

    md_path, json_path, xlsx_path = write_reports(reports, top_n=top_n)
    log.info("Wrote %s, %s, and %s", md_path, json_path, xlsx_path)

    if send_tg:
        summary = telegram_summary(reports, top_n=min(top_n, 10))
        sent = send_message(summary)
        if sent:
            send_document(str(md_path), caption=f"Full daily report — {datetime.now():%Y-%m-%d}")
            log.info("Telegram delivery: OK")
        else:
            log.info("Telegram delivery: skipped/failed")

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["watchlist", "broad"], default=RUN_MODE)
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args()
    try:
        run(mode=args.mode, top_n=args.top, send_tg=not args.no_telegram)
    except KeyboardInterrupt:
        sys.exit(1)
