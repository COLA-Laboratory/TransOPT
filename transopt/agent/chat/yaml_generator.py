from pathlib import Path
from typing import Any, Dict

import yaml
from transopt.utils.log import logger
from agent.chat.openai_chat import Message, OpenAIChat


def get_prompt(file_name: str) -> str:
    """Reads a prompt from a file."""
    current_dir = Path(__file__).parent
    file_path = current_dir / file_name
    
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt


def parse_response(response: str) -> Dict[str, Any]:
    """Parses a string response into a structured Python dictionary."""
    try:
        structured_info = yaml.safe_load(response)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing response into Python dict: {e}")
        structured_info = {}
    return structured_info


def main():
    # Assuming OpenAIChat and Message are defined elsewhere and imported correctly
    openai_chat = OpenAIChat()
    
    print("Welcome to the YAML Generator!")
    user_input = input("\nPlease describe the configuration you'd like to convert to YAML:\n")
    
    # Process the input using the OpenAI API
    prompt = get_prompt("prompt")  # Assuming the prompt file is named 'prompt.yml'
    system_message = Message(role="system", content=prompt)
    user_message = Message(role="user", content=user_input)
    response_content = openai_chat.get_response([system_message, user_message])
    
    print("\nAssistant's Response:\n")
    print(response_content)

    while True:
        refine = input("\nPlease refine your configuration or type 'exit' to quit:\n")
        if refine.lower() == 'exit':
            print("Thank you for using the YAML Generator!")
            break
        
        user_message = Message(role="user", content=refine)
        response_content = openai_chat.get_response([user_message])
        
        print("\nAssistant's Response:\n")
        print(response_content)


if __name__ == "__main__":
    main()