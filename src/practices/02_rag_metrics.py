import mlflow
import os
from dotenv import load_dotenv
from src.rag.config import Config

load_dotenv()

# Cấu hình OpenRouter
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

# ✅ Sử dụng MLflow scorers với OpenRouter
def evaluate_with_openrouter():
    from mlflow.genai.scorers import (
        answer_correctness, 
        faithfulness, 
        relevance,
        answer_relevancy,
        context_relevancy
    )
    
    # Dataset đánh giá (có thể mở rộng thêm nhiều mẫu)
    eval_data = [
        {
            "query": "What is MLflow and what is it used for?",
            "response": "MLflow is a platform for managing the machine learning lifecycle.",
            "ground_truth": "MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment.",
            "context": [
                "MLflow is an open-source platform, created by Databricks, for managing the end-to-end machine learning lifecycle.",
                "It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models."
            ]
        },
        {
            "query": "How does MLflow help with model deployment?",
            "response": "MLflow helps deploy models to various platforms.",
            "ground_truth": "MLflow provides tools to deploy models to cloud platforms, Kubernetes, or as REST APIs.",
            "context": [
                "MLflow Models allows you to package models in a standard format.",
                "You can deploy MLflow models to local servers, cloud platforms like AWS SageMaker, Azure ML, or Kubernetes clusters."
            ]
        }
    ]
    
    with mlflow.start_run() as run:
        # Đánh giá với các scorers
        results = mlflow.genai.evaluate(
            data=eval_data,
            # Không cần model vì scorers tự dùng API key
            scorers=[
                faithfulness(),        # Đánh giá tính trung thực (so với context)
                answer_correctness(),   # Độ chính xác (so với ground truth)
                relevance(),            # Tính liên quan
                answer_relevancy(),     # Mức độ liên quan của câu trả lời
                context_relevancy()     # Mức độ liên quan của context
            ],
            # Có thể cấu hình model cho scorer nếu muốn
            extra_model=f"openrouter:/{Config.MODEL_NAME}"
        )
        
        # In kết quả
        print("=" * 50)
        print("📊 EVALUATION RESULTS")
        print("=" * 50)
        
        # Lấy metrics trung bình
        metrics = results.metrics
        print(f"✅ Faithfulness Score: {metrics.get('faithfulness/v1', 0):.3f}")
        print(f"✅ Answer Correctness: {metrics.get('answer_correctness/v1', 0):.3f}")
        print(f"✅ Relevance Score: {metrics.get('relevance/v1', 0):.3f}")
        print(f"✅ Answer Relevancy: {metrics.get('answer_relevancy/v1', 0):.3f}")
        print(f"✅ Context Relevancy: {metrics.get('context_relevancy/v1', 0):.3f}")
        print("=" * 50)
        
        # In chi tiết từng sample
        print("\n📝 DETAILED RESULTS PER SAMPLE:")
        for i, row in enumerate(results.tables["eval_results_table"]):
            print(f"\nSample {i+1}:")
            print(f"  Question: {row['query']}")
            print(f"  Response: {row['response']}")
            print(f"  Faithfulness: {row.get('faithfulness/v1', 0):.3f}")
            print(f"  Correctness: {row.get('answer_correctness/v1', 0):.3f}")
            print(f"  Relevance: {row.get('relevance/v1', 0):.3f}")
        
        return results

# ✅ Cách đơn giản hơn nếu chỉ muốn test nhanh
def quick_test_with_openrouter():
    from mlflow.metrics.genai import faithfulness
    
    # Test với 1 câu hỏi
    context = "MLflow is an open-source platform, created by Databricks, for managing the end-to-end machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models."
    response = "MLflow is a platform for managing the machine learning lifecycle."
    
    # Tạo faithfulness metric
    faithfulness_metric = faithfulness()
    
    # Đánh giá
    result = faithfulness_metric.evaluate(
        response=response,
        context=context
    )
    
    print("=" * 40)
    print("🔍 QUICK FAITHFULNESS TEST")
    print("=" * 40)
    print(f"📝 Response: {response}")
    print(f"📚 Context: {context[:100]}...")
    print(f"⭐ Score: {result.score:.3f}")
    print(f"💡 Explanation: {result.justification}")
    print("=" * 40)
    
    return result

# ✅ Hàm chính
def main():
    print("🚀 Starting evaluation with OpenRouter...\n")
    
    # Test nhanh trước
    quick_test_with_openrouter()
    
    print("\n" + "=" * 50)
    print("📊 FULL EVALUATION")
    print("=" * 50)
    
    # Đánh giá đầy đủ
    results = evaluate_with_openrouter()
    
    return results

if __name__ == "__main__":
    main()