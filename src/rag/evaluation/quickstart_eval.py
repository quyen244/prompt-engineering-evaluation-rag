import mlflow
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import requests
import asyncio
from src.config import Config
import sys

from typing import Literal

# # Kiểm tra API key
# if not Config.verify_api_key():
#     sys.exit("Error: OPENROUTER_API_KEY không hợp lệ hoặc chưa cấu hình")

@dataclass
class OpenRouter:
    api_base: str
    api_key: str
    max_retries: int = 3
    max_timeout: int = 30

    async def create_chat(self, messages: List[Dict[str, Any]], model: str = "openai/gpt-3.5-turbo" , max_tokens : int = 128):
        """Tạo chat với OpenRouter API"""
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            'max_tokens' : max_tokens,
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        requests.post,
                        url=url,
                        headers=headers,
                        json=payload,
                        timeout=self.max_timeout
                    ),
                    timeout=self.max_timeout
                )

                if response.status_code == 200:
                    return response.json()

                # 429/5xx: OpenRouter free tier hay bị rate-limit upstream -> retry
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"API Error: {response.status_code} - {response.text}"
                    await asyncio.sleep(2 ** attempt)
                    continue

                raise Exception(f"API Error: {response.status_code} - {response.text}")

            except asyncio.TimeoutError as e:
                last_error = f"Timeout after {self.max_timeout}s"
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request failed: {last_error}") from e
                await asyncio.sleep(2 ** attempt)

        raise Exception(f"Request failed: {last_error}")

# Set environment
# MLflow judge chạy qua provider "openrouter" -> chỉ cần OPENROUTER_API_KEY.
# KHÔNG set OPENAI_API_KEY bằng key sk-or-... vì judge sẽ gọi thẳng api.openai.com và fail.
os.environ['OPENROUTER_API_KEY'] = Config.OPENROUTER_API_KEY
os.environ.pop('OPENAI_API_KEY', None)

# Model URI cho mọi judge/scorer của MLflow: "<provider>:/<model-name>"
JUDGE_MODEL_URI = f"openrouter:/{Config.JUDGE_MODEL}"
JUDGE_PARAMS = {"max_tokens": Config.JUDGE_MAX_TOKENS}

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('Evaluation Quickstart')

# Khởi tạo client
client = OpenRouter(
    api_base=Config.OPENROUTER_BASE_URL,
    api_key=Config.OPENROUTER_API_KEY,
    max_retries=3,
    max_timeout=30
)

def my_agent(question: str) -> str:
    """Agent đơn giản để trả lời câu hỏi"""
    messages = [
         {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
        },
        {"role": "user", "content": question},
    ]
    
    # Chạy async trong sync function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(
            client.create_chat(messages, model=Config.MODEL)
        )

        return response['choices'][0]['message']['content']
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@mlflow.trace
def qa_predict_fn(question: str) -> str:
    """Wrapper function for evaluation using ``my_agent``."""
    return my_agent(question)


eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]

from mlflow.genai import scorer
from mlflow.genai.judges import make_judge

# Evaluation with Custom Code-based Metrics
@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    # outputs có thể là None khi predict_fn lỗi (rate limit, timeout...)
    if not outputs:
        return False
    return len(outputs.split()) <= 5
    


faithfulness_judge = make_judge(
    name="faithfulness",
    instructions=(
        "You are an expert evaluator. Assess the faithfulness of the agent's response. "
        "An answer is faithful if it is fully grounded in the provided context and does not introduce any hallucinated information. "
        "Use the following inputs for evaluation: "
        "Question: {{ inputs['question'] }}, "
        "Agent's Answer: {{ outputs }}, "
        "Expected Answer (Ground Truth): {{ expectations['expected_response'] }}. "
        "Rate the faithfulness on a scale from 0.0 to 1.0, where 1.0 is completely faithful and 0.0 is completely unfaithful."
    ),
    # base_url/extra_headers KHÔNG cần: provider "openrouter" đã có sẵn endpoint
    # https://openrouter.ai/api/v1/chat/completions và tự đọc OPENROUTER_API_KEY.
    model=JUDGE_MODEL_URI,
    inference_params=JUDGE_PARAMS,
    feedback_value_type=float # Giá trị trả về là float
)


# NOTE: không dùng builtin scorer Correctness()/Guidelines() của mlflow 3.15:
# chúng không truyền `inference_params` xuống judge, nên OpenRouter nhận
# max_tokens mặc định (= context tối đa của model) và trả lỗi
# "This request requires more credits, or fewer max_tokens".
# make_judge() có truyền, nên tự định nghĩa 2 judge tương đương.
correctness_judge = make_judge(
    name="correctness",
    instructions=(
        "You are an expert evaluator. Decide whether the agent's answer is correct. "
        "Question: {{ inputs['question'] }}, "
        "Agent's Answer: {{ outputs }}, "
        "Expected Answer (Ground Truth): {{ expectations['expected_response'] }}. "
        "Answer true if the agent's answer matches the expected answer in meaning, otherwise false."
    ),
    model=JUDGE_MODEL_URI,
    inference_params=JUDGE_PARAMS,
    feedback_value_type=bool,
)

is_english_judge = make_judge(
    name="is_english",
    instructions=(
        "Check the language of the agent's answer. "
        "Agent's Answer: {{ outputs }}. "
        "Answer true if the answer is written in English, otherwise false."
    ),
    model=JUDGE_MODEL_URI,
    inference_params=JUDGE_PARAMS,
    feedback_value_type=bool,
)


scorers = [
    correctness_judge,
    is_english_judge,
    is_concise,
    faithfulness_judge,
]


if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )
    print(results)
