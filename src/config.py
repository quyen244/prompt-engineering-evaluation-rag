from dotenv import load_dotenv
import os 
import requests 

load_dotenv()

class Config:
    POROJECT_ROOT = './'
    OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY' , '')
    MODEL = 'inclusionai/ling-3.0-flash'
    # Model dùng cho LLM-as-a-judge (MLflow scorers). Tách riêng khỏi MODEL
    # của agent để có thể đổi sang model mạnh hơn khi tài khoản đủ credit.
    JUDGE_MODEL = os.getenv('JUDGE_MODEL', MODEL)
    # OpenRouter từ chối request nếu max_tokens vượt số credit còn lại, nên
    # phải gửi max_tokens tường minh cho judge.
    JUDGE_MAX_TOKENS = int(os.getenv('JUDGE_MAX_TOKENS', '128'))

    @classmethod
    def verify_api_key(cls):
        url = f"{cls.OPENROUTER_BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {cls.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": cls.MODEL,
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            'max_tokens' : 200
        }
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:

                print("\n=== Success ===")
                return True
               
            else:
                print(f"\n=== Error ===")
                print(f"Error details: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

        