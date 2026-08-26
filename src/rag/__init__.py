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
    api_key=OPENROUTER_API_KEY, 
    max_retries=2,
    max_tokens=256,
    kwargs={'thinking': False}
)
Settings.llm = llm

# Test LLM trước khi query
print("Testing LLM...")
test_response = llm.complete("Hello, say 'Hello World'")
print(f"Test response: {test_response}")


# Load documents
documents = SimpleDirectoryReader(
    r'D:\Projects\Assignment\LLM-AI-Assistant-Projects\Prompt Engineering & Evaluation\data'
).load_data()

print(f"Loaded {len(documents)} documents")



# Kiểm tra nội dung từng document
for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
    print(f"  Metadata: {doc.metadata}")
    print(f"  Text length: {len(doc.text)} characters")
    print(f"  Text preview: {doc.text[:200]}...")

# 5. Kiểm tra embedding
print("\nTesting embedding...")
try:
    embedding = Settings.embed_model.get_text_embedding("test text")
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  Embedding sample: {embedding[:5]}")
except Exception as e:
    print(f"  ❌ Embedding error: {e}")

# 6. Create index với debug
print("\nCreating index...")
try:
    index = VectorStoreIndex.from_documents(documents)
    print("  ✅ Index created successfully")
    print(f"  Index nodes: {len(index.docstore.docs)}")
except Exception as e:
    print(f"  ❌ Index creation error: {e}")
    exit()

# 7. Query
print("\nQuerying...")
query_engine = index.as_query_engine()

try:
    response = query_engine.query('what is llm? and can you explain attention in one line?')
    print(f"Response: {response}")
    print(f"Response type: {type(response)}")
    print(f"Response source nodes: {response.source_nodes if hasattr(response, 'source_nodes') else 'N/A'}")
except Exception as e:
    print(f"❌ Query error: {e}")