"""Create reproducible plots from MLflow-tracked vLLM measurement records."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/vllm_benchmark_clean/measurements.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--final-optimization", type=Path,
                        default=Path("artifacts/vllm_final_optimization"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    latest = {(r["experiment"], r["scenario"]): r for r in records}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.grid": True, "grid.alpha": .25,
                         "font.size": 10, "savefig.bbox": "tight"})
    paths = []

    concurrency = sorted((r for (exp, _), r in latest.items() if exp == "concurrency"),
                         key=lambda r: r["concurrency"])
    if concurrency:
        x = [r["concurrency"] for r in concurrency]
        fig, left = plt.subplots()
        right = left.twinx()
        left.plot(x, [r["summary"]["request_tps"] for r in concurrency], "o-", color="#0b6e75", label="Request/s")
        right.plot(x, [r["summary"]["latency_p95_s"] for r in concurrency], "s-", color="#a4442f", label="p95 latency")
        left.set(xlabel="Concurrent requests", ylabel="Completed requests/s", title="Concurrency: throughput and p95 latency")
        right.set_ylabel("p95 latency (s)")
        left.set_xticks(x)
        paths.append(args.output / "vllm_concurrency.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    prefix = [r for (exp, _), r in latest.items() if exp == "prefix"]
    if prefix:
        names = ["unique", "system_shared", "doc_shared", "partial_shared", "reordered"]
        names = [n for n in names if ("prefix", n + "-cold") in latest and ("prefix", n + "-warm") in latest]
        fig, ax = plt.subplots()
        x = range(len(names))
        ax.bar([i - .18 for i in x], [latest[("prefix", n + "-cold")]["summary"]["ttft_mean_s"] for n in names], .36, label="Cold")
        ax.bar([i + .18 for i in x], [latest[("prefix", n + "-warm")]["summary"]["ttft_mean_s"] for n in names], .36, label="Warm")
        ax.set(xlabel="Prefix structure", ylabel="Mean TTFT (s)", title="Prefix reuse: cold and warm TTFT")
        ax.set_xticks(list(x), [n.replace("_", "\n") for n in names])
        ax.legend()
        paths.append(args.output / "vllm_prefix.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    context = sorted((r for (exp, _), r in latest.items() if exp == "context" and r["concurrency"] == 1),
                     key=lambda r: int(r["scenario"].split("-")[1]))
    if context:
        x = [int(r["scenario"].split("-")[1]) for r in context]
        fig, ax = plt.subplots()
        ax.plot(x, [r["summary"]["ttft_mean_s"] for r in context], "o-", color="#0b6e75")
        ax.set(xlabel="Target context characters", ylabel="Mean TTFT (s)", title="Context size and first-token latency")
        ax.set_xticks(x)
        paths.append(args.output / "vllm_context.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    composition_names = [f"type-{i}" for i in range(4)] + ["mixed"]
    composition = [latest[("composition", name)] for name in composition_names
                   if ("composition", name) in latest]
    if composition:
        fig, ax = plt.subplots()
        labels = [r["scenario"] for r in composition]
        ax.bar(labels, [r["summary"]["latency_p95_s"] for r in composition], color="#a4442f")
        ax.set(xlabel="Workload type", ylabel="p95 latency (s)", title="Workload composition: tail latency")
        paths.append(args.output / "vllm_composition.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    stress = sorted((r for (exp, _), r in latest.items() if exp == "stress"),
                    key=lambda r: r["concurrency"])
    if stress:
        fig, left = plt.subplots()
        right = left.twinx()
        x = [r["concurrency"] for r in stress]
        left.plot(x, [r["summary"]["request_tps"] for r in stress], "o-", color="#0b6e75")
        right.plot(x, [r["summary"]["ttft_mean_s"] for r in stress], "s-", color="#a4442f")
        left.set(xlabel="Concurrent requests", ylabel="Completed requests/s", title="Stress: throughput and TTFT")
        right.set_ylabel("Mean TTFT (s)")
        left.set_xticks(x)
        paths.append(args.output / "vllm_stress.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    rag_names = ["same-document", "different-documents", "long-context"]
    rag = [latest[("rag", name)] for name in rag_names if ("rag", name) in latest]
    if rag:
        fig, left = plt.subplots()
        right = left.twinx()
        x = list(range(len(rag)))
        left.bar(x, [r["summary"]["ttft_mean_s"] for r in rag], color="#0b6e75")
        right.plot(x, [100 * r["summary"]["prefix_hit_ratio"] for r in rag], "o-", color="#a4442f")
        left.set(xlabel="RAG prompt structure", ylabel="Mean TTFT (s)", title="RAG prompts: first-token latency and cache")
        right.set_ylabel("Prefix cache hit (%)")
        left.set_xticks(x, [r["scenario"].replace("-", "\n") for r in rag])
        paths.append(args.output / "vllm_rag.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    trials = sorted(args.final_optimization.glob("*.json")) if args.final_optimization.exists() else []
    final_run_ids = {}
    if trials:
        trial = json.loads(trials[-1].read_text(encoding="utf-8"))
        layouts = ["document_first", "question_first"]
        final_run_ids = trial["run_ids"]
        fig, ax = plt.subplots()
        x = range(len(layouts))
        ax.bar([i - .18 for i in x], [trial["summary"][n]["cold_ttft_mean_s"] for n in layouts],
               .36, color="#a4442f", label="First question (cold)")
        ax.bar([i + .18 for i in x], [trial["summary"][n]["followup_ttft_mean_s"] for n in layouts],
               .36, color="#0b6e75", label="Follow-up on same document")
        ax.set(xlabel="Prompt layout", ylabel="Mean TTFT (s)",
               title="Final configuration: document-first prompt layout")
        ax.set_xticks(list(x), [n.replace("_", "\n") for n in layouts])
        ax.legend()
        paths.append(args.output / "vllm_final_optimization.svg")
        fig.savefig(paths[-1])
        plt.close(fig)

    combined = {"record_count": len(records), "latest_run_ids": {f"{exp}/{name}": r["run_id"]
                                                                for (exp, name), r in latest.items()},
                "final_optimization_run_ids": final_run_ids,
                "charts": [str(path) for path in paths]}
    summary_path = args.input.parent / "plot_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("techqa-vllm-inference")
    with mlflow.start_run(run_name="benchmark-figures") as run:
        mlflow.log_param("record_count", len(records))
        mlflow.log_dict(combined, "analysis/plot_summary.json")
        for path in paths:
            mlflow.log_artifact(str(path), "analysis/figures")
        print(json.dumps({"run_id": run.info.run_id, **combined}))


if __name__ == "__main__":
    main()
