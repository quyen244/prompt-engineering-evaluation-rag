from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
    TokenTextSplitter
)
from llama_index.core import Settings

class ChunkerType:
    SENTENCE = "sentence"
    TOKEN = "token"
    SEMANTIC = "semantic"
    WINDOW = "window"

class ChunkerFactory:
    @staticmethod
    def create_chunker(chunker_type : str , **kwargs):
        if chunker_type == ChunkerType.SENTENCE:
            return SentenceSplitter(
                **kwargs
            )
        elif chunker_type == ChunkerType.TOKEN:
            return TokenTextSplitter(
                **kwargs
            )
        elif chunker_type == ChunkerType.WINDOW:
            return SentenceWindowNodeParser(
                  **kwargs
            )
        elif chunker_type == ChunkerType.SEMANTIC:
            return SemanticSplitterNodeParser(
                **kwargs
            )
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")



    