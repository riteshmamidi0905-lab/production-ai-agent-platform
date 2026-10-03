"""CLI: `python -m agentplatform.cli ingest|eval|serve`. `--embedded-pg` uses a throw-away local PostgreSQL+pgvector (pgserver, dev extra)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from .config import Settings
from .container import Container


async def _with_container(args, fn):
    settings = Settings()
    server = None
    if args.embedded_pg:
        import pgserver
        server = pgserver.get_server(tempfile.mkdtemp())
        settings = Settings(database_url=server.get_uri())
    c = await Container.build(settings)
    try:
        return await fn(c)
    finally:
        await c.close()
        if server:
            server.cleanup()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentplatform")
    ap.add_argument("--embedded-pg", action="store_true", help="use a temporary embedded PostgreSQL (needs the dev extra)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest"); ing.add_argument("path", nargs="?", default="corpus")
    ev = sub.add_parser("eval")
    ev.add_argument("--provider", default="offline"); ev.add_argument("--repeats", type=int, default=2); ev.add_argument("--corpus", default="corpus")
    ev.add_argument("--gate", action="store_true", help="exit 1 if metrics regress against the stored baseline")
    ev.add_argument("--update-baseline", action="store_true"); ev.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    async def go() -> int:
        async def run(c: Container) -> int:
            if args.cmd == "ingest":
                print(json.dumps(await c.ingestor.ingest_dir(args.path), indent=2)); return 0
            from .eval.runner import gate, load_baseline, run_eval
            await c.ingestor.ingest_dir(args.corpus)
            result = await run_eval(c, args.provider, repeats=args.repeats)
            g = gate(result["metrics"], load_baseline(args.provider))
            result["gate"] = g
            if args.update_baseline:
                from importlib import resources
                p = Path(str(resources.files("agentplatform.eval"))) / "baselines" / f"{args.provider}.json"
                p.write_text(json.dumps({"provider": args.provider, "dataset": result["dataset"], "metrics": result["metrics"]}, indent=2) + "\n")
                print("baseline written:", p)
            if args.out:
                Path(args.out).write_text(json.dumps(result, indent=2))
            print(json.dumps({"metrics": result["metrics"], "gate": g}, indent=2))
            return 0 if (g["passed"] or not args.gate) else 1
        return await _with_container(args, run)
    return asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())
