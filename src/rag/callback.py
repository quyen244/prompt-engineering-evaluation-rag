# # src/rag/optimization/main.py

# class TokenTracker:
#     """Theo dõi token usage trong toàn bộ optimization"""
    
#     def __init__(self):
#         self.total_prompt_tokens = 0
#         self.total_completion_tokens = 0
#         self.total_tokens = 0
#         self.total_cost = 0.0
#         self.token_usage_per_question = {}
        
#     def track(self, question: str, usage: dict):
#         """Track token usage cho mỗi question"""
#         prompt = usage.get('prompt_tokens', 0)
#         completion = usage.get('completion_tokens', 0)
#         total = usage.get('total_tokens', 0)
#         cost = usage.get('cost', 0.0)
        
#         self.total_prompt_tokens += prompt
#         self.total_completion_tokens += completion
#         self.total_tokens += total
#         self.total_cost += cost
        
#         # Lưu chi tiết
#         self.token_usage_per_question[question[:30]] = {
#             'prompt_tokens': prompt,
#             'completion_tokens': completion,
#             'total_tokens': total
#         }
        
#     def log_to_mlflow(self):
#         """Log tất cả token metrics vào MLflow"""
#         with mlflow.start_run(run_name="token_summary", nested=True):
#             mlflow.log_metrics({
#                 "total_prompt_tokens": self.total_prompt_tokens,
#                 "total_completion_tokens": self.total_completion_tokens,
#                 "total_tokens_all": self.total_tokens,
#                 "total_cost": self.total_cost,
#                 "avg_tokens_per_question": self.total_tokens / len(self.token_usage_per_question) if self.token_usage_per_question else 0
#             })
            
#             # Log chi tiết từng question
#             for q, usage in list(self.token_usage_per_question.items())[:10]:  # Chỉ log 10 sample
#                 mlflow.log_metrics({
#                     f"tokens_{q[:20]}": usage['total_tokens']
#                 })
                
#     def print_summary(self):
#         """In summary ra console"""
#         print("\n" + "="*50)
#         print("📊 TOKEN USAGE SUMMARY")
#         print("="*50)
#         print(f"Total Prompt Tokens: {self.total_prompt_tokens:,}")
#         print(f"Total Completion Tokens: {self.total_completion_tokens:,}")
#         print(f"Total Tokens: {self.total_tokens:,}")
#         print(f"Total Cost: ${self.total_cost:.6f}")
#         print(f"Avg Tokens/Question: {self.total_tokens / len(self.token_usage_per_question):.0f}")
#         print("="*50)



# # src/rag/optimization/main.py

# class TokenTracker:
#     """Theo dõi token usage trong toàn bộ optimization"""
    
#     def __init__(self):
#         self.total_prompt_tokens = 0
#         self.total_completion_tokens = 0
#         self.total_tokens = 0
#         self.total_cost = 0.0
#         self.token_usage_per_question = {}
        
#     def track(self, question: str, usage: dict):
#         """Track token usage cho mỗi question"""
#         prompt = usage.get('prompt_tokens', 0)
#         completion = usage.get('completion_tokens', 0)
#         total = usage.get('total_tokens', 0)
#         cost = usage.get('cost', 0.0)
        
#         self.total_prompt_tokens += prompt
#         self.total_completion_tokens += completion
#         self.total_tokens += total
#         self.total_cost += cost
        
#         # Lưu chi tiết
#         self.token_usage_per_question[question[:30]] = {
#             'prompt_tokens': prompt,
#             'completion_tokens': completion,
#             'total_tokens': total
#         }
        
#     def log_to_mlflow(self):
#         """Log tất cả token metrics vào MLflow"""
#         with mlflow.start_run(run_name="token_summary", nested=True):
#             mlflow.log_metrics({
#                 "total_prompt_tokens": self.total_prompt_tokens,
#                 "total_completion_tokens": self.total_completion_tokens,
#                 "total_tokens_all": self.total_tokens,
#                 "total_cost": self.total_cost,
#                 "avg_tokens_per_question": self.total_tokens / len(self.token_usage_per_question) if self.token_usage_per_question else 0
#             })
            
#             # Log chi tiết từng question
#             for q, usage in list(self.token_usage_per_question.items())[:10]:  # Chỉ log 10 sample
#                 mlflow.log_metrics({
#                     f"tokens_{q[:20]}": usage['total_tokens']
#                 })
                
#     def print_summary(self):
#         """In summary ra console"""
#         print("\n" + "="*50)
#         print("📊 TOKEN USAGE SUMMARY")
#         print("="*50)
#         print(f"Total Prompt Tokens: {self.total_prompt_tokens:,}")
#         print(f"Total Completion Tokens: {self.total_completion_tokens:,}")
#         print(f"Total Tokens: {self.total_tokens:,}")
#         print(f"Total Cost: ${self.total_cost:.6f}")
#         print(f"Avg Tokens/Question: {self.total_tokens / len(self.token_usage_per_question):.0f}")
#         print("="*50)