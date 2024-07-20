from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv
load_dotenv('.env')

class TogetherAiClass:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        together_api_key = os.getenv("TOGETHER_API_KEY")
        self.llm = TogetherLLM(model=model_name, api_key=together_api_key)
        print("LLM initialized")

    def complete_query(self, query):
        resp = self.llm.complete(query)
        print(resp)

    def chat_with_llm(self, system_message="You are a pirate with a colorful personality", query="What is your name"):
        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=query),
        ]
        resp = self.llm.chat(messages)
        print(resp)

    def stream_complete_query(self, query):
        response = self.llm.stream_complete(query)
        for r in response:
            print(r.delta, end="")

    def stream_chat_with_llm(self, system_message="You are a pirate with a colorful personality", query="What is your name"):
        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=query),
        ]
        resp = self.llm.stream_chat(messages)
        for r in resp:
            print(r.delta, end="")

    def interactive_chat(self, system_message):
        system_message = system_message
        messages = [ChatMessage(role="system", content=system_message)]    
        print("Welcome to the interactive pirate chat! Type 'quit' to exit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                print("Arrr, farewell matey!")
                break
            
            messages.append(ChatMessage(role="user", content=user_input))

            print("\nPirate:", end=" ")
            response_content = ""
            for chunk in self.llm.stream_chat(messages):
                print(chunk.delta, end="", flush=True)
                response_content += chunk.delta
            
            messages.append(ChatMessage(role="assistant", content=response_content))
            
            if len(messages) > 10: 
                messages = messages[:1] + messages[-9:]  # Keep system message and last 9 exchanges

#
# model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
model_name = "meta-llama/Llama-3-70b-chat-hf"
tai_obj = TogetherAiClass(model_name)
tai_obj.interactive_chat("You are a pirate with a colorful personality")
