# src/app_streamlit.py
import streamlit as st
from pathlib import Path
from src.rag.rag_chatbot import get_chatbot
from src.rag.config import Config

# Cấu hình page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 RAG Chatbot")
st.caption("Chat with your documents using LlamaIndex + OpenRouter")

# Khởi tạo chatbot (chỉ chạy 1 lần)
@st.cache_resource
def init_chatbot():
    PROJECT_ROOT = Path(__file__).parent.parent
    data_dir = PROJECT_ROOT / "data"
    
    return get_chatbot(
        data_dir=str(data_dir),
        chunk_size=200,
        chunk_overlap=50,
        model_name=Config.MODEL_NAME
    )

try:
    chatbot = init_chatbot()
    st.success("✅ Chatbot ready!")
except Exception as e:
    st.error(f"❌ Failed to initialize chatbot: {e}")
    st.stop()

# Hiển thị số documents
with st.expander("📊 Document Info"):
    if chatbot.documents:
        st.write(f"📄 Documents: {len(chatbot.documents)}")
        for i, doc in enumerate(chatbot.documents[:3], 1):
            st.write(f"  {i}. {doc.metadata.get('file_name', 'Unknown')}")
        if len(chatbot.documents) > 3:
            st.write(f"  ... and {len(chatbot.documents) - 3} more")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chatbot.chat(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selector
    model_options = [
        "openrouter/anthropic/claude-3-haiku",
        "openrouter/anthropic/claude-3.5-sonnet",
        "openrouter/meta-llama/llama-3.1-70b-instruct",
        "openrouter/google/gemini-1.5-flash"
    ]
    
    selected_model = st.selectbox("Model", model_options)
    if selected_model != chatbot.model_name:
        st.warning("Model change requires restart")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Made with ❤️ using LlamaIndex")

# Footer
st.divider()
st.caption("💡 Tip: You can ask questions about the content in your documents")