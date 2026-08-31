# src/rag/eval_dataset.py
import json
from pathlib import Path
from typing import List, Dict

class EvalDataset:
    """Tạo và quản lý dataset cho evaluation"""
    
    @staticmethod
    def create_qa_dataset():
        """Tạo bộ câu hỏi - đáp án mẫu từ documents"""
        
        # Đây là dataset mẫu, bạn nên thu thập từ người dùng thực tế
        questions = [
            {
                "question": "What is Artificial Intelligence?",
                "expected_answer": "AI is the simulation of human intelligence processes by computer systems.",
                "context": "Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction."
            },
            {
                "question": "What is Machine Learning?",
                "expected_answer": "Machine Learning is a subset of AI that enables systems to learn from experience without being explicitly programmed.",
                "context": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves."
            },
            {
                "question": "What is Deep Learning?",
                "expected_answer": "Deep Learning is a specialized branch of machine learning that uses neural networks with multiple layers.",
                "context": "Deep Learning is a specialized branch of machine learning that uses neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain, allowing it to learn from large amounts of data."
            },
            {
                "question": "What is Natural Language Processing?",
                "expected_answer": "NLP is an area of AI that deals with the interaction between computers and humans using natural language.",
                "context": "Natural Language Processing (NLP) is another important area of AI that deals with the interaction between computers and humans using natural language. The ultimate goal of NLP is to read, decipher, understand, and make sense of human languages."
            },
            {
                "question": "What is Computer Vision?",
                "expected_answer": "Computer Vision is a field of AI that trains computers to interpret and understand the visual world.",
                "context": "Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos, deep learning models can accurately identify and classify objects."
            }
        ]
        
        # Lưu dataset
        eval_dir = Path("eval_data")
        eval_dir.mkdir(exist_ok=True)
        
        file_path = eval_dir / "qa_dataset.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created evaluation dataset: {file_path}")
        return questions
    
    @staticmethod
    def load_qa_dataset() -> List[Dict]:
        """Load dataset từ file"""
        file_path = Path("eval_data/qa_dataset.json")
        
        if not file_path.exists():
            return EvalDataset.create_qa_dataset()
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == '__main__':
    EvalDataset.create_qa_dataset()


    dataset = EvalDataset.load_qa_dataset()

    print(dataset)