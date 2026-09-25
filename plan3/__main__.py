from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ROOT


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    f=sub.add_parser("features")
    f.add_argument("--prepared",type=Path,default=ROOT/"work/plan3/prepared")
    f.add_argument("--views",type=Path,default=ROOT/"work/plan3/views")
    f.add_argument("--indexes",type=Path,default=ROOT/"work/plan3/indexes")
    f.add_argument("--run",type=Path,required=True)
    f.add_argument("--query-batch",type=int,default=250)
    f.add_argument("--word-k",type=int)
    f.add_argument("--missing-k",type=int)
    t=sub.add_parser("train")
    t.add_argument("--prepared",type=Path,default=ROOT/"work/plan3/prepared")
    t.add_argument("--run",type=Path,required=True)
    t.add_argument("--threads",type=int,default=4)
    t.add_argument("--max-rounds",type=int,default=6000)
    s=sub.add_parser("support")
    s.add_argument("--prepared",type=Path,default=ROOT/"work/plan3/prepared")
    s.add_argument("--run",type=Path,required=True)
    s.add_argument("--threads",type=int,default=4)
    s.add_argument("--max-rounds",type=int,default=6000)
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
    elif args.command == "features":
        from .featurize import featurize
        featurize(args.prepared,args.views,args.indexes,args.run,args.query_batch,args.word_k,args.missing_k)
    elif args.command == "train":
        from .train import train_direct
        train_direct(args.prepared,args.run,args.threads,args.max_rounds)
    elif args.command == "support":
        from .support import train_support
        train_support(args.prepared,args.run,args.threads,args.max_rounds)


if __name__ == "__main__":
    main()
