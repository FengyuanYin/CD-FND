"""CD-FND 多智能体假新闻检测系统入口(科研实验 CLI)。

用法示例:
    # 训练/验证闭环(默认只跑基线,不调用 Optimizer)
    python main.py --dataset weibo21 --mode train --epochs 1 --batch-size 8

    # 让 Optimizer 每个批次出诊断与候选建议(不改任何文件)
    python main.py --dataset weibo21 --mode train --epochs 1 --optimization suggest

    # 完整优化闭环:候选在独立验证集上通过才晋升,失败自动回滚
    python main.py --dataset weibo21 --mode train --epochs 3 --optimization apply

    # 冻结测试域的最终评估(不优化、不改 Skill)
    python main.py --dataset weibo21 --mode test

密钥只从环境变量读取(审计 3.7 节):
    set DEEPSEEK_API_KEY=sk-...   (Windows) / export DEEPSEEK_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from config import API_KEY_ENV, Config
from trainner import Trainer, run_test_evaluation


def _build_config(args: argparse.Namespace) -> Config:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(
            f"警告:未设置环境变量 {API_KEY_ENV},真实运行将失败;"
            "请不要把密钥写进代码。",
            file=sys.stderr,
        )
    return Config(
        model_name=os.environ.get("DEEPSEEK_MODEL", args.model),
        base_url=args.base_url,
        api_key=api_key,
        epoch=args.epochs,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        optimization_mode=args.optimization,
        max_validation_items=args.max_validation_items,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CD-FND misinformation-detection multi-agent experiments"
    )
    parser.add_argument("--dataset", default="weibo21",
                        choices=["weibo", "weibo21", "amtcele"])
    parser.add_argument("--mode", default="train", choices=["train", "test"],
                        help="train=训练/验证闭环;test=冻结测试域最终评估")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimization", default="off",
                        choices=["off", "suggest", "apply"],
                        help="off=基线; suggest=只出建议; apply=验证门控晋升")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1,
                        help="训练域内切给 meta-validation 的比例")
    parser.add_argument("--max-validation-items", type=int, default=8,
                        help="候选 Skill 验证最多评测的样本数")
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--run-id", default=None,
                        help="本次运行的目录名;缺省自动按时间生成")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _build_config(args)

    if args.mode == "train":
        trainer = Trainer(config)
        logs = asyncio.run(trainer.train())
        print(f"训练完成,批次日志 {len(logs)} 条。输出目录: {trainer.run_dir}")
        for entry in logs:
            metrics = entry.get("metrics", {})
            print(
                f"  epoch {entry['epoch']} [{entry['category']}] "
                f"batch {entry['batch_index']} n={entry['batch_size']} "
                f"acc={metrics.get('accuracy')} macro_f1={metrics.get('macro_f1')} "
                f"mode={entry.get('optimization', {}).get('mode')}"
            )
    else:
        summary = asyncio.run(run_test_evaluation(config))
        metrics = summary["metrics"]
        print(f"测试评估完成(n={summary['n_items']}):")
        print(f"  accuracy={metrics.get('accuracy')} macro_f1={metrics.get('macro_f1')}")
        print(f"  summary: {summary['run_dir']}/summary.json")


if __name__ == "__main__":
    main()
