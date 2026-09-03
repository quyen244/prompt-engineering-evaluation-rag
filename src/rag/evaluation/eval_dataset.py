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
        eval_dataset = [
            {
                "inputs": {"question": "What is the capital of France?"},
                "expectations": {"expected_response": "Paris"},
            },
            {
                "inputs": {"question": "Who was the first person to build an airplane?"},
                "expectations": {"expected_response": "Wright Brothers"},
            },
            {
                "inputs": {"question": "Who wrote Romeo and Juliet?"},
                "expectations": {"expected_response": "William Shakespeare"},
            },
        ]
            
        # Lưu dataset
        eval_dir = Path("eval_data")
        eval_dir.mkdir(exist_ok=True)
        
        file_path = eval_dir / "qa_dataset.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(eval_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created evaluation dataset: {file_path}")
        return eval_dataset
    
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