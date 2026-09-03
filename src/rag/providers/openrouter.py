from typing import Dict , List , Any
import asyncio
import requests
from dataclasses import dataclass
from src.config import Config
import json 
import mlflow 


# Model URI cho mọi judge/scorer của MLflow: "<provider>:/<model-name>"
JUDGE_MODEL_URI = f"openrouter:/{Config.JUDGE_MODEL}"
JUDGE_PARAMS = {"max_tokens": Config.JUDGE_MAX_TOKENS, "thinking": False}


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
            'thinking' : False,
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



# Khởi tạo client
client = OpenRouter(
    api_base=Config.OPENROUTER_BASE_URL,
    api_key=Config.OPENROUTER_API_KEY,
    max_retries=3,
    max_timeout=30
)


def my_agent(question: str, prompt_template : str = None) -> str:
    """Agent đơn giản để trả lời câu hỏi"""

    if prompt_template:
        messages = [
             {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions concisely and shortly",
            },
            {"role": "user", "content": prompt_template}
        ]
    else:
        # Default prompt (fallback)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely and shortly under 20 words.",
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

        print(f"📝 Response keys: {response.keys() if isinstance(response, dict) else type(response)}")
        if isinstance(response, dict):
            print(f"📝 Response keys: {response.keys()}")
            
               # Kiểm tra lỗi từ OpenRouter
            if 'error' in response:
                print(f"❌ OpenRouter Error: {json.dumps(response['error'], indent=2)}")
                # Thử parse lỗi chi tiết
                if isinstance(response['error'], dict):
                    error_msg = response['error'].get('message', str(response['error']))
                    raise Exception(f"OpenRouter API Error: {error_msg}")
                else:
                    raise Exception(f"OpenRouter API Error: {response['error']}")
             
            # Mọi thứ OK
            content = response['choices'][0]['message']['content']
            print(f"✅ Response content: {content[:50]}...")  # In 50 ký tự đầu
            
            # Lấy token usage từ response
            if 'usage' in response:
                usage = response['usage']
                
                # === LOG TOKEN METRICS VÀO MLFLOW ===
                with mlflow.start_run(nested=True):  # Nested run để theo dõi
                    mlflow.log_metrics({
                        "prompt_tokens": usage.get('prompt_tokens', 0),
                        "completion_tokens": usage.get('completion_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0),
                        "cost": usage.get('cost', 0),
                        
                        # Thêm chi tiết nếu có
                        "cached_tokens": usage.get('prompt_tokens_details', {}).get('cached_tokens', 0),
                    })
                    
                    # Log thêm tags
                    mlflow.set_tags({
                        "model": response.get('model', 'unknown'),
                        "provider": response.get('provider', 'unknown'),
                        "service_tier": response.get('service_tier', 'free')
                    })
                    
            if 'model' in response:
                print(f"🤖 Model used: {response['model']}")
            
            return content
        
    
        return response['choices'][0]['message']['content']
    except asyncio.TimeoutError as e:
        print(f"⏰ Timeout error: {e}")
        raise Exception(f"Request timeout after {client.max_timeout}s") from e
        
    except Exception as e:
        print(f"❌ Exception in my_agent: {type(e).__name__}: {e}")
        # Log thêm context
        print(f"Question: {question[:50]}...")
        print(f"Model: {Config.MODEL}")
        raise

    finally:
        loop.close()
        asyncio.set_event_loop(None)

    


