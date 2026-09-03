from mlflow.genai.optimize import GepaPromptOptimizer
from src.rag.scorers import correctness_judge
import mlflow
from pathlib import Path
from src.rag.providers.openrouter import my_agent, JUDGE_MODEL_URI
import argparse
import os 
from src.config import Config


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
prompt_path = current_path.parent / 'prompts'

def registry_prompt(prompt_name: str):
    with open(prompt_path / (prompt_name + '.txt'), 'r', encoding='utf-8') as f:
        template = f.read()
    prompt = mlflow.genai.register_prompt(
        name='qa-agent-1',
        template=template
    )
    return prompt

@mlflow.trace
def qa_pred_fn(question: str):
    return my_agent(question)

def create_dataset():
    # ============ TẠO TRAIN DATASET ============
    train_dataset = [
        # === CÂU HỎI CƠ BẢN VỀ HIV ===
        {
            "inputs": {"question": "HIV có phải là bệnh mãn tính không?"},
            "expectations": {"expected_response": "Có, HIV là bệnh mãn tính."}
        },
        {
            "inputs": {"question": "HIV lây truyền qua những con đường nào?"},
            "expectations": {"expected_response": "HIV lây qua đường máu, tình dục và từ mẹ sang con."}
        },
        {
            "inputs": {"question": "HIV có thể lây qua đường ăn uống không?"},
            "expectations": {"expected_response": "Không, HIV không lây qua đường ăn uống hay tiếp xúc thông thường."}
        },
        {
            "inputs": {"question": "HIV có chữa khỏi được không?"},
            "expectations": {"expected_response": "Hiện chưa có thuốc chữa khỏi HIV nhưng có thể kiểm soát bằng ARV."}
        },
        {
            "inputs": {"question": "ARV là gì?"},
            "expectations": {"expected_response": "ARV là thuốc kháng retrovirus dùng điều trị HIV."}
        },
        
        # === CHẨN ĐOÁN VÀ XÉT NGHIỆM ===
        {
            "inputs": {"question": "Xét nghiệm HIV có chính xác không?"},
            "expectations": {"expected_response": "Xét nghiệm HIV có độ chính xác cao, đặc biệt sau 3 tháng phơi nhiễm."}
        },
        {
            "inputs": {"question": "Khi nào nên xét nghiệm HIV?"},
            "expectations": {"expected_response": "Nên xét nghiệm sau 2-4 tuần và xác nhận sau 3 tháng kể từ phơi nhiễm."}
        },
        {
            "inputs": {"question": "Xét nghiệm HIV ở đâu?"},
            "expectations": {"expected_response": "Xét nghiệm tại bệnh viện, trung tâm y tế hoặc cơ sở tư vấn xét nghiệm tự nguyện."}
        },
        
    ]

    # Kiểm tra số lượng
    print(f"✅ Đã tạo {len(train_dataset)} mẫu train")

    # Xem một vài mẫu
    for i, sample in enumerate(train_dataset[:3]):
        print(f"\n📝 Sample {i+1}:")
        print(f"   Question: {sample['inputs']['question']}")
        print(f"   Expected: {sample['expectations']['expected_response']}")

    return train_dataset

def main():
    parser = argparse.ArgumentParser(description='Optimize prompt with GEPA')
    parser.add_argument('--prompt_name', required=True, help='Tên file prompt')
    parser.add_argument('--reflection_model', default = JUDGE_MODEL_URI,
                       help='Model dùng để phân tích và sinh prompt mới')
    parser.add_argument('--max_metric_calls', default = 5,
                           help='maximum of evolving turns')
    args = parser.parse_args()

    print("\n[Step 1] Registering prompt...")
    prompt = registry_prompt(args.prompt_name)

    # Step 2: Load train data
    print("\n[Step 2] Loading train dataset...")
    train_dataset = create_dataset()
    print(f"✅ Loaded {len(train_dataset)} training samples")

    # Step 3: Optimize!
    print("\n[Step 3] Optimizing with GEPA...")
    print(f"   Reflection model: {args.reflection_model}")
    
    with mlflow.start_run(run_name=f"GEPA_{args.prompt_name}"):
        result = mlflow.genai.optimize_prompts(
            predict_fn=qa_pred_fn,
            train_data=train_dataset,
            prompt_uris=[prompt.uri],
            optimizer=GepaPromptOptimizer(
                reflection_model=args.reflection_model,
                max_metric_calls=5,
                display_progress_bar=True
            ),
            scorers=[correctness_judge],
            enable_tracking=True
        )
    
    # Step 4: Show results
    print("\n[Step 4] Optimization complete!")
    optimized_prompt = result.optimized_prompts[0]
    
    print("\n" + "="*40 + " OPTIMIZED PROMPT " + "="*40)
    print(optimized_prompt.template)
    print("="*96)
    
    # Log best prompt
    print(f"\n✅ Best prompt URI: {optimized_prompt.uri}")
    print(f"✅ View results at: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    main()