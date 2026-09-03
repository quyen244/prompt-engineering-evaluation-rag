import mlflow
import os
from dataclasses import dataclass
from src.rag.providers.openrouter import my_agent
from src.config import Config
import json 
from src.rag.scorers import faithfulness_judge , correctness_judge , is_concise
# Set environment
# MLflow judge chạy qua provider "openrouter" -> chỉ cần OPENROUTER_API_KEY.
# KHÔNG set OPENAI_API_KEY bằng key sk-or-... vì judge sẽ gọi thẳng api.openai.com và fail.
os.environ['OPENROUTER_API_KEY'] = Config.OPENROUTER_API_KEY
os.environ.setdefault('MLFLOW_GENAI_EVAL_MAX_WORKERS', '1')



# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('Evaluation Quickstart Version 3')


@mlflow.trace
def qa_predict_fn(question: str) -> str:
    """Wrapper function for evaluation using ``my_agent``."""
    return my_agent(question)

def main():
    try:
    # dataset 
        with open(f'D:\Projects\Assignment\LLM-AI-Assistant-Projects\Prompt Engineering & Evaluation\eval_data\qa_dataset.json' , 'r' , encoding='utf-8') as f:
            eval_dataset = json.load(f)

        if isinstance(eval_dataset , dict):
            print('loaded dataset')

    except Exception as e:
        raise e


    scorers = [
        correctness_judge,
        is_concise,
        faithfulness_judge,
    ]

    results = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=qa_predict_fn,
            scorers=scorers,
        )
    print(results)


if __name__ == "__main__":
    SystemExit(main())
    
