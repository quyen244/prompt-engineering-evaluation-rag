import sys 
from pathlib import Path
from src.rag.rag_chatbot import get_chatbot
from src.rag.config import Config

def main():
    """CLI Chat Interface"""
    print("=" * 60)
    print("🤖 RAG CHATBOT")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"

    print(f"\n📂 Loading data from: {data_dir}")
    chatbot = get_chatbot(
        data_dir=str(data_dir),
        chunk_size=200,
        chunk_overlap=50,
        model_name=Config.MODEL_NAME
    )

    print("\n✅ Chatbot ready! Type 'exit' to quit.\n")
    print("-" * 60)


    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if user_input in ['exit' , 'quit' , 'bye']:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Gửi câu hỏi
            print("🤔 Thinking...")
            response = chatbot.chat(user_input)

            print(f"\n🤖 Bot: {response}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()


# Q1 : does it autimatically retrive when chatting ? 


# 1. Về câu hỏi: "Does it automatically retrieve when chatting?"
# Có! Khi bạn dùng chat_engine.chat(), nó tự động:

# ✅ Retrieve các chunks liên quan từ index

# ✅ Generate response dựa trên context đã retrieve

# ✅ Memory lưu lịch sử hội thoại (nếu dùng ChatMemoryBuffer)

# python
# # Khi bạn gọi:
# response = self.chat_engine.chat("What is AI?")

# Quy trình tự động:
# 1. Retrieve: Tìm các chunks liên quan đến câu hỏi
# 2. Generate: Gửi context + question lên LLM
# 3. Response: Trả về answer


# system prompt (assistant role) + condensed history chat + retrieved context & question (user role)



