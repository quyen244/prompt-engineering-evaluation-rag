from llama_index.core import SimpleDirectoryReader
from pathlib import Path
from src.rag.chunking import ChunkerFactory
from llama_index.embeddings.openai import OpenAIEmbedding , OpenAIEmbeddingModelType
from dotenv import load_dotenv
import os 

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# project_root = Path(__file__).parent.parent.parent
# input_dir = project_root / "data"

# print(input_dir)
# docs = SimpleDirectoryReader(input_dir=input_dir).load_data()


# # Khi bạn dùng SimpleDirectoryReader, mỗi file trong thư mục sẽ được chuyển thành 1 Document object.


# # docs là 1 List[Document]
# print(type(docs))        # <class 'list'>
# print(len(docs))         # Số lượng file
# print(type(docs[0]))     # <class 'llama_index.core.schema.Document'>



# # Mỗi Document có các thuộc tính chính:
# doc = docs[0]

# # 1. Nội dung văn bản
# print(doc.text)           # Nội dung thô của file
# print(doc.text[:100])     # 100 ký tự đầu

# # 2. Metadata (thông tin file)
# print(doc.metadata)
# # Ví dụ output:
# # {
# #     'file_path': './knowledge_base/report.pdf',
# #     'file_name': 'report.pdf',
# #     'file_type': 'application/pdf',
# #     'file_size': 1024576,
# #     'creation_date': '2024-01-15',
# #     'last_modified_date': '2024-01-20'
# # }

# # 3. ID của document
# print(doc.doc_id)         # UUID tự sinh hoặc bạn có thể set

# # 4. Embedding (nếu đã được tạo)
# print(doc.embedding)      # List[float] hoặc None nếu chưa embed



# from llama_index.core.node_parser import SentenceSplitter

# splitter = SentenceSplitter(chunk_size = 512 , chunk_overlap= 64)
# nodes = splitter.get_nodes_from_documents(docs)



# # nodes là 1 List[TextNode]
# print(type(nodes))        # <class 'list'>
# print(len(nodes))         # Số chunk sau khi chia
# print(type(nodes[0]))     # <class 'llama_index.core.schema.TextNode'>

# print('node' + 50 * '=')

# node = nodes[0]

# # 1. Nội dung văn bản của chunk này
# print('node text : ' , node.text)
# # Ví dụ: "This is the first chunk of text from the document..."

# # 2. Metadata (được kế thừa từ Document gốc + thêm thông tin chunk)
# print(node.metadata)
# # {
# #     'file_path': './knowledge_base/report.pdf',
# #     'file_name': 'report.pdf',
# #     'page_label': '1',
# #     'chunk_index': 0,           # Thêm vào
# #     'chunk_total': 15,          # Thêm vào
# #     'chunk_size': 512
# # }

# # 3. ID của node (unique)
# print(node.node_id)       # UUID

# # 4. Thông tin vị trí trong document gốc
# print(node.start_char_idx)   # Vị trí ký tự bắt đầu
# print(node.end_char_idx)     # Vị trí ký tự kết thúc

# # 5. Embedding (nếu đã được tạo)
# print(node.embedding)     # List[float] hoặc None


PROJECT_ROOT = Path(__file__).parent.parent.parent
input_dir = PROJECT_ROOT  / "data"

# Đọc file
documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
doc = documents[0]  

# # Thêm debug trước khi chạy semantic splitter
# print(f"Document type: {type(doc)}")
# print(f"Has doc_id: {hasattr(doc, 'doc_id')}")
# print(f"doc_id value: {getattr(doc, 'doc_id', 'NO ID')}")
# print(f"Has id_: {hasattr(doc, 'id_')}")

print("=" * 80)
print(f"FILE INFO")
print("=" * 80)
print(f"File name: {doc.metadata.get('file_name')}")
print(f"Total characters: {len(doc.text)}")
print(f"Total words: {len(doc.text.split())}")
print(f"First 200 chars: {doc.text[:200]}...")
print("\n")

# ============================================================================
# CÁCH 1: SentenceSplitter - Chia theo câu
# ============================================================================
print("=" * 80)
print("CÁCH 1: SENTENCE SPLITTER (Chia theo câu)")
print("=" * 80)

sentence_splitter = ChunkerFactory.create_chunker('sentence' ,    chunk_size=200,          # Số ký tự tối đa trong 1 chunk
    chunk_overlap=50,        # Chồng lấp giữa các chunk
    separator=" ",           # Dấu phân cách
    paragraph_separator="\n",)

nodes_sentence = sentence_splitter.get_nodes_from_documents([doc])

print(f"Number of chunks: {len(nodes_sentence)}")
print(f"Chunk size: 200 characters")
print(f"Chunk overlap: 50 characters")



# ============================================================================
# CÁCH 2: TokenTextSplitter - Chia theo token
# ============================================================================
print("\n" + "=" * 80)
print("CÁCH 2: TOKEN TEXT SPLITTER (Chia theo token)")
print("=" * 80)

token_splitter = ChunkerFactory.create_chunker('token' , 
                                        chunk_size=100,          # Số tokens tối đa trong 1 chunk
                                        chunk_overlap=20,        # Chồng lấp token
                                        separator=" ",
                                        backup_separators=["\n", ";"],
                                    )

nodes_token = token_splitter.get_nodes_from_documents([doc])
print(f"Number of chunks: {len(nodes_token)}")
print(f"Chunk size: 100 tokens")
print(f"Chunk overlap: 20 tokens")

# ============================================================================
# CÁCH 3: SemanticSplitterNodeParser - Chia theo ngữ nghĩa (cần embedding)
# ============================================================================
print("\n" + "=" * 80)
print("CÁCH 3: SEMANTIC SPLITTER (Chia theo ngữ nghĩa)")
print("=" * 80)

embed_model = OpenAIEmbedding(model = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
                            api_key=OPENROUTER_API_KEY , 
                            api_base='https://openrouter.ai/api/v1')


semantic_splitter = ChunkerFactory.create_chunker('semantic' ,
                                                embed_model=embed_model,
                                                buffer_size=1,           # Số câu xung quanh để xác định ngữ nghĩa
                                                breakpoint_percentile_threshold=95
                                                  # Ngưỡng % để cắt
    )

print("Đang phân tích ngữ nghĩa... (có thể mất vài giây)")

try:
    nodes_semantic = semantic_splitter.get_nodes_from_documents([doc])
    
    print(f"Number of chunks: {len(nodes_semantic)}")
    print(f"Chunk sizes vary based on semantic similarity")
    print("\nCONTENT OF EACH CHUNK:")
    print("-" * 40)
    
    for i, node in enumerate(nodes_semantic, 1):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(node.text)} characters, {len(node.text.split())} words")
        print(f"  Preview: {node.text[:150]}...")
        
except Exception as e:
    print(f"Error with semantic splitter: {e}")


# ============================================================================
# SO SÁNH 3 CÁCH
# ============================================================================
print("\n" + "=" * 80)
print("SO SÁNH 3 CÁCH CHUNKING")
print("=" * 80)

print(f"""
┌─────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Đặc điểm            │ Sentence Splitter        │ Token Splitter           │ Semantic Splitter        │
├─────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Số chunks           │ {len(nodes_sentence):<24}│ {len(nodes_token):<24}   │ {len(nodes_semantic) if 'nodes_semantic' in locals() else 'N/A':<24}│
│ Dựa trên            │ Câu và ký tự             │ Token (words)            │ Ngữ nghĩa của văn bản   │
│ Tốc độ              │ Nhanh                    │ Rất nhanh                │ Chậm (cần embedding)    │
│ Giữ nguyên câu      │ ✅ Có                   │ ❌ Có thể cắt giữa câu  │ ✅ Có                   │
│ Giữ nguyên ý nghĩa  │ Khá tốt                 │ Trung bình               │ ✅ Rất tốt             │
│ Phù hợp             │ Văn bản có cấu trúc     │ Văn bản ngắn, code      │ Văn bản dài, phức tạp  │
└─────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘
""")
