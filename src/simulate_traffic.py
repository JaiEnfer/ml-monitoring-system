from __future__ import annotations

import random
import time

import requests

API = "http://127.0.0.1:8000"


def send_payload(payload: dict) -> None:
    r = requests.post(f"{API}/predict", json=payload, timeout=10)
    r.raise_for_status()


def baseline_payload() -> dict:
    # Similar to training distribution
    return {
        "age": random.uniform(25, 45),
        "income": random.uniform(35000, 75000),
        "years_employed": random.uniform(1, 10),
        "credit_score": random.uniform(600, 780),
    }


def shifted_payload() -> dict:
    # Intentionally shifted distribution (should trigger drift)
    return {
        "age": random.uniform(55, 70),
        "income": random.uniform(120000, 200000),
        "years_employed": random.uniform(15, 35),
        "credit_score": random.uniform(300, 520),
    }


def run(mode: str = "baseline", n: int = 200, sleep_ms: int = 10) -> None:
    generator = baseline_payload if mode == "baseline" else shifted_payload
    for i in range(n):
        send_payload(generator())
        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)
        if (i + 1) % 50 == 0:
            print(f"Sent {i+1}/{n} {mode} events")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "shifted"], default="baseline")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sleep-ms", type=int, default=10)
    args = parser.parse_args()

    run(mode=args.mode, n=args.n, sleep_ms=args.sleep_ms)
    print("Done.")
