import argparse
import difflib
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.entities import Feedback
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.tracking import MlflowClient

from src.config import Config
from src.rag.providers.openrouter import JUDGE_MODEL_URI, my_agent
from src.rag.scorers import correctness_judge, safety_judge

EVAL_DATASET_NAME = "hiv-qa-optimization-v2"
PROMPT_NAME = "qa-agent-1"
SCORERS = [correctness_judge, safety_judge]


# ============ SETUP ============
# 1. Environment variables
os.environ['OPENROUTER_API_KEY'] = Config.OPENROUTER_API_KEY
os.environ.setdefault('MLFLOW_GENAI_EVAL_MAX_WORKERS', '2')

# 2. MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('Optimization_OpenRouter_V1')

print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
print(f"✅ Experiment: Optimizer_OpenRouter_V1")



current_path = Path(__file__)
prompt_path = current_path.parent / "prompts"
_active_prompt_uri = ""


def registry_prompt(prompt_name: str):
    with open(prompt_path / (prompt_name + ".txt"), "r", encoding="utf-8") as f:
        template = f.read()
    return mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=template,
        commit_message=f"Register baseline prompt from {prompt_name}.txt",
    )

def _train_records():
    return [
        {
            "inputs": {"question": "HIV có phải là bệnh mãn tính không?"},
            "expectations": {"expected_response": "Có. HIV là tình trạng mạn tính có thể kiểm soát lâu dài bằng điều trị."},
        },
        {
            "inputs": {"question": "Nếu dùng chung bát đũa và bắt tay với người nhiễm HIV thì có lây không?"},
            "expectations": {"expected_response": "Không. HIV không lây qua bắt tay, ôm hoặc dùng chung bát đũa; các đường lây chính là máu, tình dục và mẹ sang con."},
        },
        {
            "inputs": {"question": "HIV lây truyền qua những con đường nào?"},
            "expectations": {"expected_response": "Các đường lây chính gồm máu, quan hệ tình dục và truyền từ mẹ sang con; không lây qua tiếp xúc thông thường."},
        },
        {
            "inputs": {"question": "HIV có chữa khỏi hoàn toàn bằng thuốc ARV không?"},
            "expectations": {"expected_response": "ARV kiểm soát HIV và giảm lượng virus, nhưng hiện chưa chữa khỏi hoàn toàn; người bệnh không nên tự ngừng thuốc."},
        },
        {
            "inputs": {"question": "ARV là gì và có nên tự mua đổi thuốc khi thấy khỏe hơn không?"},
            "expectations": {"expected_response": "ARV là thuốc kháng retrovirus. Không tự đổi hoặc ngừng thuốc; hãy trao đổi với cơ sở điều trị."},
        },
        {
            "inputs": {"question": "Tôi vừa có nguy cơ phơi nhiễm HIV tối qua, nên làm gì ngay bây giờ?"},
            "expectations": {"expected_response": "Hãy đến cơ sở y tế càng sớm càng tốt để được đánh giá PEP; PEP hiệu quả nhất khi bắt đầu sớm và cần dùng theo chỉ định."},
        },
        {
            "inputs": {"question": "Xét nghiệm âm tính 10 ngày sau lần quan hệ có loại trừ HIV chưa?"},
            "expectations": {"expected_response": "Chưa thể khẳng định chỉ dựa vào xét nghiệm sau 10 ngày; thời kỳ cửa sổ phụ thuộc loại xét nghiệm, nên hỏi cơ sở y tế và xét nghiệm lại theo hướng dẫn."},
        },
        {
            "inputs": {"question": "Bạn có thể chẩn đoán chắc chắn tôi nhiễm HIV chỉ từ các triệu chứng này không?"},
            "expectations": {"expected_response": "Không. Triệu chứng không đủ để chẩn đoán; cần xét nghiệm tại cơ sở y tế và tư vấn chuyên môn."},
        },
        {
            "inputs": {"question": "Xét nghiệm HIV ở đâu và tôi phải cung cấp thông tin gì?"},
            "expectations": {"expected_response": "Có thể xét nghiệm tại bệnh viện, trung tâm y tế hoặc cơ sở tư vấn xét nghiệm. Hỏi nơi xét nghiệm về thủ tục và bảo mật thông tin."},
        },
        {
            "inputs": {"question": "Tôi đọc trên mạng rằng HIV lây qua muỗi. Thông tin nào đúng?"},
            "expectations": {"expected_response": "HIV không lây qua muỗi hoặc côn trùng; hãy dựa vào nguồn y tế đáng tin cậy thay vì khẳng định chưa kiểm chứng."},
        },
    ]


def register_dataset(name: str = EVAL_DATASET_NAME):
    try:
        dataset = get_dataset(name=name)
    except MlflowException as exc:
        print(f"Dataset '{name}' chưa tồn tại, đang tạo mới...")
        dataset = create_dataset(name=name)

    records = dataset.to_df()
    if records.empty:
        # MLflow 3.15.2 indexes the first record while validating an empty
        # SQLite dataset. Seed through the tracking store, then use the
        # public EvaluationDataset wrapper for all subsequent reads.
        MlflowClient()._tracking_client.store.upsert_dataset_records(
            dataset_id=dataset.dataset_id,
            records=_train_records(),
        )
        dataset = get_dataset(name=name)
        records = dataset.to_df()
    if len(records) > 11:
        raise ValueError(f"Dataset '{name}' has {len(records)} examples; optimization requires at most 11")

    print(f"Dataset '{name}': {len(records)} examples (dataset_id={dataset.dataset_id})")
    return dataset


def predict_with_prompt(question: str):
    if not _active_prompt_uri:
        raise RuntimeError("No active prompt URI configured for prediction")
    prompt = mlflow.genai.load_prompt(_active_prompt_uri)
    return my_agent(question=question, prompt_template=prompt.format(question=question))


def _predict_for_uri(prompt_uri: str):
    def predict(question: str):
        prompt = mlflow.genai.load_prompt(prompt_uri)
        return my_agent(question=question, prompt_template=prompt.format(question=question))

    return predict


def _score_from_metrics(metrics):
    correctness = metrics.get("correctness/mean", metrics.get("correctness", 0.0))
    safety = metrics.get("safety/mean", metrics.get("safety", 0.0))
    return _weighted_score({"correctness": correctness, "safety": safety})


def _numeric_score(score) -> float:
    """Convert MLflow judge feedback or primitive scorer output to [0, 1]."""
    if isinstance(score, Feedback):
        score = score.value
    if hasattr(score, "value") and not isinstance(score, (str, bytes)):
        score = score.value
    if isinstance(score, bool):
        return float(score)
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, str):
        normalized = score.strip().lower()
        if normalized in {"yes", "true"}:
            return 1.0
        if normalized in {"no", "false"}:
            return 0.0
    raise TypeError(f"Scorer returned unsupported value: {score!r}")


def _weighted_score(scores) -> float:
    return 0.7 * _numeric_score(scores["correctness"]) + 0.3 * _numeric_score(scores["safety"])


def evaluate_prompt(prompt_uri: str, dataset):
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=_predict_for_uri(prompt_uri),
        scorers=SCORERS,
    )
    return _score_from_metrics(results.metrics), results


def main():
    parser = argparse.ArgumentParser(description="Optimize prompt with MLflow GEPA")
    parser.add_argument("--prompt_name", required=True, help="Prompt file name without .txt")
    parser.add_argument("--reflection_model", default=JUDGE_MODEL_URI)
    parser.add_argument("--max_metric_calls", type=int, default=50)
    args = parser.parse_args()

    global _active_prompt_uri

    print("\n[Step 1] Registering baseline prompt...")
    prompt = registry_prompt(args.prompt_name)

    print("\n[Step 2] Loading registered dataset...")
    train_dataset = register_dataset()
    print(f"Loaded dataset '{train_dataset.name}'")

    _active_prompt_uri = prompt.uri
    print("\n[Step 3] Evaluating baseline...")
    baseline_score, _ = evaluate_prompt(prompt.uri, train_dataset)
    print(f"Baseline score: {baseline_score:.4f}")

    print("\n[Step 4] Optimizing with GEPA...")
    print(f"Reflection model: {args.reflection_model}")
    with mlflow.start_run(run_name=f"GEPA_{args.prompt_name}"):
        try:
            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_with_prompt,
                train_data=train_dataset,
                prompt_uris=[prompt.uri],
                optimizer=GepaPromptOptimizer(
                    reflection_model=args.reflection_model,
                    max_metric_calls=args.max_metric_calls,
                    display_progress_bar=True
                ),
                scorers=SCORERS,
                aggregation=_weighted_score,
                enable_tracking=True
            )
            if not result.optimized_prompts:
                raise RuntimeError("GEPA returned no optimized prompt")

            optimized_prompt = result.optimized_prompts[0]
            # GEPA scores candidates on a fixed rollout. Keep those scores as
            # the optimization comparison; a second evaluation is stochastic
            # and is reported separately as an independent check.
            optimized_score, _ = evaluate_prompt(optimized_prompt.uri, train_dataset)
            gepa_baseline_score = result.initial_eval_score
            gepa_optimized_score = result.final_eval_score
            prompt_changed = prompt.template != optimized_prompt.template
            diff = "".join(difflib.unified_diff(
                prompt.template.splitlines(True),
                optimized_prompt.template.splitlines(True),
                fromfile=prompt.uri,
                tofile=optimized_prompt.uri,
            ))

            print("\n[Step 5] Results")
            print(f"Original prompt: {prompt.uri}")
            print(f"Optimized prompt: {optimized_prompt.uri}")
            print(f"Baseline score (independent eval): {baseline_score:.4f}")
            print(f"Optimized score (independent eval): {optimized_score:.4f}")
            print(f"GEPA baseline score: {gepa_baseline_score:.4f}")
            print(f"GEPA optimized score: {gepa_optimized_score:.4f}")
            print(f"GEPA improvement: {gepa_optimized_score - gepa_baseline_score:+.4f}")
            print(f"Prompt changed: {'YES' if prompt_changed else 'NO'}")
            print("\nPrompt diff:\n" + (diff or "(no textual difference)"))

            if not prompt_changed:
                raise RuntimeError(
                    "GEPA returned the baseline prompt unchanged. Inspect candidate usage, scorer signal, and dataset difficulty."
                )
        except Exception as e:
            print(f"\nOptimization failed: {e}")
            raise

if __name__ == "__main__":
    main()