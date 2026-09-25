from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ROOT


def main():
    parser = argparse.ArgumentParser(description="Plan 3 reproducible entity-resolution experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, default=ROOT / "student_resource/dataset")
    prep.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    prep.add_argument("--audit-size", type=int, default=50_000)
    prep.add_argument("--include-test", action="store_true")
    ret = sub.add_parser("retrieve")
    ret.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    ret.add_argument("--indexes", type=Path, default=ROOT / "work/plan3/indexes")
    ret.add_argument("--run", type=Path, required=True)
    ret.add_argument("--profile", choices=["probe", "pilot", "full"], default="probe")
    ret.add_argument("--channels", default="word,missing")
    ret.add_argument("--threads", type=int, default=4)
    ret.add_argument("--query-batch", type=int, default=1000)
    ret.add_argument("--shard-size", type=int, default=50_000)
    v = sub.add_parser("views")
    v.add_argument("--prepared",type=Path,default=ROOT/"work/plan3/prepared")
    v.add_argument("--out",type=Path,default=ROOT/"work/plan3/views")
    v.add_argument("--split",choices=["train","test"],default="train")
    b = sub.add_parser("budget")
    b.add_argument("--prepared",type=Path,default=ROOT/"work/plan3/prepared")
    b.add_argument("--run",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.dataset, args.prepared, args.audit_size, args.include_test)
    elif args.command == "retrieve":
        from .retrieve import DEFAULT_BUDGETS, retrieve
        budgets = {ch: DEFAULT_BUDGETS[ch] for ch in args.channels.split(",")}
        retrieve(args.prepared, args.indexes, args.run, args.profile, budgets, args.threads,
                 args.query_batch, args.shard_size)
    elif args.command == "views":
        from .views import prepare_views
        prepare_views(args.prepared,args.out,args.split)
    elif args.command == "budget":
        from .budget import select_budget
        select_budget(args.prepared,args.run)


if __name__ == "__main__":
    main()
