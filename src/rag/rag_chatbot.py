import os 
from pathlib import Path
from typing import List , Optional


from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex , 
    Settings, 
    Document
)


from src.rag.chunking import ChunkerFactory
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.openai import OpenAIEmbedding , OpenAIEmbeddingModelType
# chat
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer



from dotenv import load_dotenv

from src.rag.config import Config

load_dotenv()

class RAGChatBot:
    """RAG ChatBot with LLamaIndex"""

    def __init__(self,
                 data_dir : Optional[str] = None,
                 chunk_size : int = 200,
                 chunk_overlap : int = 50,
                 model_name : str = Config.MODEL_NAME,
                 max_tokens : int = 512):
        """Khởi tạo chatbot
        
        Args:
            data_dir: Đường dẫn đến thư mục chứa documents
            chunk_size: Kích thước chunk
            chunk_overlap: Chồng lấp giữa các chunk
            model_name: Tên model OpenRouter
            max_tokens: Số token tối đa cho response
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.max_tokens = max_tokens


        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
                    raise ValueError("OPENROUTER_API_KEY not found in .env file")


        self._setup_embedding()

        self._setup_llm()

        if data_dir:
              self.documents = self._load_documents(data_dir)
              self.index = self._build_index(documents=self.documents)
              self._setup_chat_engine()

        else:
              pass

    def _setup_embedding(self):
        Settings.embed_model = OpenAIEmbedding(
              api_base="https://openrouter.ai/api/v1",
              api_key=self.api_key,
              model = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
              max_retries=2
        )

        print("✅ Embedding model configured")

    def _setup_llm(self):
        self.llm =  OpenRouter(
              api_base="https://openrouter.ai/api/v1",
              api_key=self.api_key,
              model = self.model_name,
              max_retries=2,
              max_tokens= self.max_tokens,
              kwargs={
                    'thinking' : False
              }
          )
        
        Settings.llm = self.llm
        print(f"✅ LLM configured: {self.model_name}")

    def _load_documents(self , data_dir) -> List[Document]:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
        print(f"✅ Loaded {len(documents)} documents from {data_dir}")
        return documents


    def _build_index(self, documents : List[Document]):
        splitter = ChunkerFactory.create_chunker('sentence',
                                                   chunk_size=self.chunk_size,
                                                    chunk_overlap=self.chunk_overlap,
                                                    separator=" ",
                                                    paragraph_separator="\n")

        nodes = splitter.get_nodes_from_documents(documents)
        print(f"✅ Created {len(nodes)} chunks")
        
        # Create index
        index = VectorStoreIndex(nodes)
        print(f"✅ Index created with {len(index.docstore.docs)} nodes")
        
        return index
    
    def _setup_chat_engine(self):
          
        memory = ChatMemoryBuffer.from_defaults(
             token_limit=4000
        )

        self.chat_engine = self.index.as_chat_engine(
             chat_mode='condense_plus_context',
             memory = memory,
             system_prompt = (
                 "You are a helpful AI assistant. Answer questions based on the provided context.\n"
                "If you don't know the answer, say 'I don't have information about that.'\n"
                "Be concise and accurate."
             ),
             verbose = True
        )

        print("✅ Chat engine configured")


    def chat(self , message  : str ) -> str:
        if not self.chat_engine:
            raise ValueError("Chat engine not initialized. Please provide data_dir first.")

        try:
            response = self.chat_engine.chat(message)
            return str(response)

        except Exception as e:
              return f"Error: {str(e)}"

    def query(self, question : str) -> str:
         if not self.index:
              raise ValueError("Index not initialized")

         query_engine = self.index.as_query_engine()
         response = query_engine.query(question)

         return response

    def add_document(self, text: str, metadata: Optional[dict] = None):
        """Thêm document mới vào index"""
        if not self.index:
            raise ValueError("Index not initialized.")
        
        doc = Document(text=text, metadata=metadata or {})
        self.index.insert(doc)
        print("✅ Document added to index")
    
    def reload_index(self, data_dir: str):
        """Reload index từ thư mục mới"""
        self.documents = self._load_documents(data_dir)
        self.index = self._build_index(self.documents)
        self._setup_chat_engine()
        print("✅ Index reloaded")

        


_chatbot_instance = None


def get_chatbot(
        data_dir : Optional[str] = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        model_name: str = Config.MODEL_NAME,
        max_tokens: int = 512
):
    global _chatbot_instance

    if _chatbot_instance is None:
        _chatbot_instance = RAGChatBot(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            max_tokens=max_tokens
        )

    return _chatbot_instance
    
    



# 1. Cấu trúc message khi gửi lên LLM
# Khi bạn gọi chat_engine.chat("What is AI?"), LlamaIndex sẽ xây dựng message theo cấu trúc sau:




# Cấu trúc message cuối cùng gửi lên LLM:
# messages = [
#     {
#         "role": "system",
#         "content": """You are a helpful AI assistant. Answer questions based on the provided context.
# If you don't know the answer, say 'I don't have information about that.'
# Be concise and accurate."""
#     },
#     {
#         "role": "user",
#         "content": """Context information is below:
# ---------------------
# [Retrieved chunks from vector store]
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Query: What is AI?"""
#     }
# ]



# 2. Quy trình chi tiết từ input đến response

# Bước 1: User input
# user_message = "What is AI?"

# Bước 2: Retrieve (tìm kiếm các chunks liên quan)
# retrieved_nodes = index.retrieve("What is AI?")
# => Lấy ra các TextNode có score cao nhất

# Bước 3: Format context từ retrieved nodes
# context = "\n".join([node.text for node in retrieved_nodes])
# Ví dụ context:
# "Artificial Intelligence (AI) is the simulation of human intelligence..."
# "Machine Learning is a subset of AI that enables systems to learn..."

# Bước 4: Xây dựng messages với condense_question (nếu có history)
# Nếu là lần chat đầu tiên:

# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
# ]

# Nếu đã có history (chat_mode='condense_question'):
# Lịch sử chat sẽ được "condense" (nén) thành context
# history = [
#     {"role": "user", "content": "What is AI?"},
#     {"role": "assistant", "content": "AI is the simulation of human intelligence..."},
#     {"role": "user", "content": "Tell me more about Machine Learning"}
# ]

# => Câu hỏi mới được condense thành:
# "User asked about Machine Learning in the context of AI"

# 3. Các chat_mode trong LlamaIndex

# 1. condense_question (default - bạn đang dùng)
# self.index.as_chat_engine(chat_mode="condense_question")
# # Mỗi câu hỏi mới sẽ được nén với lịch sử
# # Ví dụ: 
# # User: "What is AI?"
# # Bot: "AI is..."
# # User: "What about ML?" 
# # => Condense thành: "What about Machine Learning in the context of AI?"

# # 2. context (giữ nguyên context)
# self.index.as_chat_engine(chat_mode="context")
# # Giữ nguyên toàn bộ lịch sử chat trong context
# # Phù hợp với model có context window lớn

# # 3. react (dùng ReAct agent)
# self.index.as_chat_engine(chat_mode="react")
# # Cho phép bot thực hiện nhiều bước reasoning

# # 4. openai (dùng OpenAI function calling)
# self.index.as_chat_engine(chat_mode="openai")
# # Dùng function calling để retrieve