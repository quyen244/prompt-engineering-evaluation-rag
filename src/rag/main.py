"""Minimal command-line entry point for the LangChain RAG foundation."""
from __future__ import annotations

import argparse

from .rag_chatbot import get_chatbot


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a directory with the RAG pipeline")
    parser.add_argument("data_dir")
    args = parser.parse_args()
    chatbot = get_chatbot(data_dir=args.data_dir)
    while True:
        question = input("question> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer = chatbot.query(question)
            print(answer.text)
            for source in answer.sources:
                print(f"  [{source.chunk_id}] {source.metadata.get('source', '<unknown>')}")


if __name__ == "__main__":
    main()
