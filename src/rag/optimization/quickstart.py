import argparse
import difflib
import os

import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer

from src.config import Config
from src.rag.providers.openrouter import JUDGE_MODEL_URI, my_agent
from src.rag.scorers import classification_accuracy

# ============ SETUP ============
os.environ["OPENROUTER_API_KEY"] = Config.OPENROUTER_API_KEY
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Optimization_Quickstart_Classification")

print(f"[SETUP] Tracking URI: {mlflow.get_tracking_uri()}")
print(f"[SETUP] Application model: {Config.MODEL}")
print(f"[SETUP] Reflection/judge model: {JUDGE_MODEL_URI}")

PROMPT_NAME = "medical_section_classifier"

# 11 câu, cân bằng 5 lớp (2/2/2/3/2), mỗi câu nhắm vào một ranh giới khó giữa
# hai nhãn liền kề để buộc prompt phải học phân biệt thay vì đoán theo từ khoá.
DATASET = [
    # BACKGROUND rõ ràng
    {
        "inputs": {
            "question": "Antiretroviral therapy has transformed HIV infection from a fatal illness into a manageable chronic condition, yet adherence barriers remain poorly understood in adolescent populations."
        },
        "expectations": {"expected_response": "BACKGROUND"},
    },
    {
        "inputs": {
            "question": "Despite widespread availability of antiretroviral therapy, retention in care during the first year after HIV diagnosis remains a major barrier to achieving viral suppression."
        },
        "expectations": {"expected_response": "BACKGROUND"},
    },
    # BACKGROUND vs OBJECTIVE: motivation + explicit stated aim -> OBJECTIVE
    {
        "inputs": {
            "question": "Although several digital adherence tools exist, none have been tested specifically among adolescents recently diagnosed with HIV, so this study aimed to evaluate the feasibility of a smartphone-based reminder app in this group."
        },
        "expectations": {"expected_response": "OBJECTIVE"},
    },
    {
        "inputs": {
            "question": "The purpose of this trial was to determine whether a nurse-led counseling intervention improves retention in HIV care during the first year after diagnosis."
        },
        "expectations": {"expected_response": "OBJECTIVE"},
    },
    # OBJECTIVE vs METHODS: intervention description + how it was measured -> METHODS
    {
        "inputs": {
            "question": "Participants received a 12-week peer-support telehealth program, and medication adherence was assessed monthly using electronic pill-bottle caps that recorded each opening."
        },
        "expectations": {"expected_response": "METHODS"},
    },
    {
        "inputs": {
            "question": "Blood samples were collected at baseline, week 12, and week 24, and CD4 counts were determined by flow cytometry at each visit."
        },
        "expectations": {"expected_response": "METHODS"},
    },
    # METHODS vs RESULTS: methodological terms + an observed outcome -> RESULTS
    {
        "inputs": {
            "question": "Viral load was measured by RT-PCR at baseline and week 24, and the intervention group showed a significantly greater decline than the control group (p<0.01)."
        },
        "expectations": {"expected_response": "RESULTS"},
    },
    {
        "inputs": {
            "question": "Mean CD4 count increased by 143 cells/mm3 in the intervention arm compared with 61 cells/mm3 in the control arm at 24 weeks."
        },
        "expectations": {"expected_response": "RESULTS"},
    },
    # RESULTS vs CONCLUSIONS: result phrased with an implication clause -> still RESULTS
    # because it reports the measured numbers, not just the takeaway.
    {
        "inputs": {
            "question": "Adherence rose from 62% to 89% after introducing the reminder app, a change not observed in the usual-care group during the same period."
        },
        "expectations": {"expected_response": "RESULTS"},
    },
    # CONCLUSIONS rõ ràng / CONCLUSIONS vs BACKGROUND: broad implication, no new data
    {
        "inputs": {
            "question": "Overall, the results support incorporating peer counseling into routine HIV care to sustain long-term adherence."
        },
        "expectations": {"expected_response": "CONCLUSIONS"},
    },
    {
        "inputs": {
            "question": "These findings suggest that community-based adherence support should be considered a standard part of HIV treatment programs in resource-limited settings."
        },
        "expectations": {"expected_response": "CONCLUSIONS"},
    },
]

SCORERS = [classification_accuracy]


def registry_prompt():
    return mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=(
            "Classify this medical research paper sentence into one of these sections: "
            "CONCLUSIONS, RESULTS, METHODS, OBJECTIVE, BACKGROUND.\n\n"
            "Sentence: {{question}}"
        ),
        commit_message="Register baseline classifier prompt",
    )


def _predict_for_uri(prompt_uri: str):
    """Trả predict_fn dùng đúng bản prompt hiện hành ứng với `prompt_uri`.

    LƯU Ý QUAN TRỌNG (đã xác minh bằng cách đọc source
    mlflow/genai/optimize/optimize.py::_build_eval_fn): trong lúc
    `optimize_prompts` chạy, MLflow monkey-patch thuộc tính
    `PromptVersion.template` ở CẤP CLASS, khớp theo `prompt.name` chứ
    không phải theo URI/version cụ thể. Nghĩa là dù load_prompt(prompt_uri)
    trỏ tới version nào, trong lúc GEPA đánh giá một candidate, `.template`
    vẫn trả về nội dung candidate đang được thử — không cần predict_fn tự
    biết "candidate hiện tại là gì". Ngoài optimizer (baseline eval trước/
    sau khi optimize), `.template` trả về đúng nội dung thật của version đó.
    """

    def predict(question: str):
        prompt = mlflow.genai.load_prompt(prompt_uri)
        formatted = prompt.format(question=question)
        print(f"[OPTIMIZATION] Q: {question[:70]!r}...")
        try:
            output = my_agent(question=question, prompt_template=formatted)
        except Exception as e:
            print(f"[ERROR] Application LLM failure: {type(e).__name__}: {e}")
            raise
        print(f"[OPTIMIZATION] -> Output: {output!r}")
        return output

    return predict


def evaluate_prompt(prompt_uri: str, label: str):
    print(f"\n[EVAL] {label}: {prompt_uri}")
    results = mlflow.genai.evaluate(
        data=DATASET,
        predict_fn=_predict_for_uri(prompt_uri),
        scorers=SCORERS,
    )
    score = results.metrics.get("classification_accuracy/mean", 0.0)
    print(f"[EVAL] {label} accuracy: {score:.4f} ({score * len(DATASET):.0f}/{len(DATASET)})")
    return score, results


def main():
    parser = argparse.ArgumentParser(description="Optimize the classifier prompt with GEPA")
    parser.add_argument("--reflection_model", default=JUDGE_MODEL_URI)
    # Baseline full-valset eval alone costs len(DATASET) metric calls (11), and every
    # rejected candidate proposal costs another reflection_minibatch_size (=3, GEPA
    # default) before a real optimization signal appears. The previous default of 5
    # was smaller than even one baseline pass, so GEPA stopped before proposing a
    # single new candidate. 200 gives room for dozens of proposal rounds.
    parser.add_argument("--max_metric_calls", type=int, default=200)
    args = parser.parse_args()

    print("\n[Step 1] Registering baseline prompt...")
    prompt = registry_prompt()
    print(f"[Step 1] Registered: {prompt.uri}")

    print("\n[Step 2] Baseline evaluation...")
    baseline_score, _ = evaluate_prompt(prompt.uri, "BASELINE")

    print("\n[Step 3] Optimizing with GEPA...")
    print(f"[Step 3] Reflection model: {args.reflection_model}")
    print(f"[Step 3] Application model: {Config.MODEL}")
    print(f"[Step 3] max_metric_calls: {args.max_metric_calls}")
    if args.reflection_model == f"openrouter:/{Config.MODEL}":
        print(
            "[Step 3][WARNING] Reflection model is the SAME model as the application "
            "model. Self-critique from a small free model is weaker than using a "
            "stronger model for reflection; set JUDGE_MODEL env var to override."
        )

    with mlflow.start_run(run_name="GEPA_quickstart_classification"):
        result = mlflow.genai.optimize_prompts(
            predict_fn=_predict_for_uri(prompt.uri),
            train_data=DATASET,
            prompt_uris=[prompt.uri],
            optimizer=GepaPromptOptimizer(
                reflection_model=args.reflection_model,
                max_metric_calls=args.max_metric_calls,
                display_progress_bar=True,
                # LiteLLM already retries transient errors (incl. ConnectionResetError)
                # 3x by default for the reflection LM; bump it slightly since OpenRouter's
                # free tier is flaky, without hiding a persistently broken reflection call.
                gepa_kwargs={"reflection_lm_kwargs": {"num_retries": 5, "timeout": 60}},
            ),
            scorers=SCORERS,
            enable_tracking=True,
        )

        if not result.optimized_prompts:
            raise RuntimeError("GEPA returned no optimized prompt")

        optimized_prompt = result.optimized_prompts[0]
        prompt_changed = prompt.template != optimized_prompt.template

        print("\n[Step 4] Optimized evaluation (independent check)...")
        optimized_score, _ = evaluate_prompt(optimized_prompt.uri, "OPTIMIZED")

        diff = "".join(
            difflib.unified_diff(
                prompt.template.splitlines(True),
                optimized_prompt.template.splitlines(True),
                fromfile=prompt.uri,
                tofile=optimized_prompt.uri,
            )
        )

        print("\n" + "=" * 20 + " RESULT " + "=" * 20)
        print(f"Baseline score (independent eval): {baseline_score:.4f}")
        print(f"Optimized score (independent eval): {optimized_score:.4f}")
        print(f"GEPA baseline score: {result.initial_eval_score:.4f}")
        print(f"GEPA optimized score: {result.final_eval_score:.4f}")
        print(
            "GEPA improvement: "
            f"{result.final_eval_score - result.initial_eval_score:+.4f} "
            f"({(result.final_eval_score - result.initial_eval_score) / max(result.initial_eval_score, 1e-9):+.1%} relative)"
        )
        print(f"Prompt changed: {'YES' if prompt_changed else 'NO'}")
        print(f"Original prompt: {prompt.uri}")
        print(f"Optimized prompt: {optimized_prompt.uri}")
        print("\nOriginal template:\n" + prompt.template)
        print("\nOptimized template:\n" + optimized_prompt.template)
        print("\nPrompt diff:\n" + (diff or "(no textual difference)"))

        if not prompt_changed:
            print(
                "\n[WARNING] GEPA returned the baseline prompt unchanged. This means "
                "no proposed candidate ever beat the baseline on its evaluation "
                "minibatch — check the [OPTIMIZATION]/[SCORE] logs above for whether "
                "the model is producing usable output, and consider raising "
                "--max_metric_calls further or using a stronger --reflection_model."
            )


if __name__ == "__main__":
    main()
