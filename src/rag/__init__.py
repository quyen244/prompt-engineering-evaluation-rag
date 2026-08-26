import os 
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openrouter import OpenRouter
from src.rag.config import Config
from llama_index.embeddings.openai import OpenAIEmbedding , OpenAIEmbeddingModelType

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY

# Dùng enum đúng cách
Settings.embed_model = OpenAIEmbedding(
    api_base='https://openrouter.ai/api/v1',
    model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,  # Hoặc model="text-embedding-ada-002"
    api_key=OPENROUTER_API_KEY,
    max_retries=2
)


# LLM
llm = OpenRouter(
    model=Config.MODEL_NAME,
    max_retries=2,
    max_tokens=256,
    kwargs={'thinking': False}
)
Settings.llm = llm

# Load documents
documents = SimpleDirectoryReader(
    r'D:\Projects\Assignment\LLM-AI-Assistant-Projects\Prompt Engineering & Evaluation\data'
).load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query('what is llm ? and can you explain attention in one line ?')
print(response)