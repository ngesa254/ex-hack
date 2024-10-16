import boto3
import json
from git import Repo
import os

# Create a Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# AWS configuration
anthropic_claude_haiku_Id = "anthropic.claude-3-haiku-20240307-v1:0"
anthropic_claude_sonnet_Id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_id = anthropic_claude_sonnet_Id

# The prompt for the AI agents
PROMPT = """
You are an advanced AI agent in the Agentic Code Recreation System. Your role is to analyze, document, and recreate a codebase from a given GitHub repository. Follow these instructions carefully:

1. Repository Analysis:
   - Analyze the structure, files, and dependencies of the cloned repository
   - Identify the main application file and its purpose
   - Determine the programming languages and frameworks used

2. Documentation Generation:
   - Create a comprehensive Markdown document (documentation.md) with the following sections:
     a. Project Overview
     b. Repository Structure
     c. Dependencies and Requirements
     d. Code Architecture
     e. Main Components and Their Functions
     f. API Endpoints (if applicable)
     g. Data Flow
     h. Usage Examples

3. Instruction Creation:
   - Generate a step-by-step guide (instructions.md) for recreating the codebase, including:
     a. Environment Setup
     b. Dependency Installation
     c. File Structure Creation
     d. Code Implementation Steps
     e. Testing Procedures

4. Code Recreation:
   - Recreate the main application file (app.py) and any necessary supporting files
   - Ensure functional equivalence with the original codebase
   - Implement best practices and maintain high code quality

Remember to maintain a high standard of code quality and documentation, ensure functional equivalence, and consider potential edge cases and error handling.
"""

def invoke_model(prompt):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    })
    response = client.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')

def clone_repository(repo_url, local_path):
    Repo.clone_from(repo_url, local_path)
    print(f"Repository cloned to {local_path}")

def read_repository_contents(repo_path):
    contents = {}
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.txt', '.md', '.json', '.yaml', '.yml')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    contents[file_path] = f.read()
    return contents

def analyze_repository(repo_path):
    contents = read_repository_contents(repo_path)
    content_str = "\n\n".join([f"File: {path}\n\nContent:\n{content}" for path, content in contents.items()])
    prompt = f"{PROMPT}\n\nAnalyze the following repository contents and provide a detailed analysis:\n\n{content_str}"
    return invoke_model(prompt)

def generate_documentation(analysis):
    prompt = f"{PROMPT}\n\nBased on this analysis, generate comprehensive documentation in Markdown format:\n\n{analysis}"
    documentation = invoke_model(prompt)
    with open("documentation.md", "w") as f:
        f.write(documentation)
    print("Documentation generated and saved as documentation.md")

def create_instructions(analysis):
    prompt = f"{PROMPT}\n\nBased on this analysis, create detailed instructions for recreating the codebase in Markdown format:\n\n{analysis}"
    instructions = invoke_model(prompt)
    with open("instructions.md", "w") as f:
        f.write(instructions)
    print("Instructions generated and saved as instructions.md")

def recreate_code(analysis, repo_path):
    contents = read_repository_contents(repo_path)
    content_str = "\n\n".join([f"File: {path}\n\nContent:\n{content}" for path, content in contents.items()])
    prompt = f"{PROMPT}\n\nBased on this analysis and the original code, recreate the main application file (app.py) and any necessary supporting files. Ensure that the recreated code is functionally equivalent to the original:\n\nAnalysis:\n{analysis}\n\nOriginal Code:\n{content_str}\n\nPlease provide the complete recreated code for app.py:"
    recreated_code = invoke_model(prompt)
    with open("app.py", "w") as f:
        f.write(recreated_code)
    print("Code recreated and saved as app.py")

def main():
    repo_url = "https://github.com/KimaniKibuthu/hackathon-iris-app.git"
    local_path = "./cloned_repo"

    print("Starting Agentic Code Recreation System")
    print("AI Prompt:", PROMPT)

    clone_repository(repo_url, local_path)
    analysis = analyze_repository(local_path)
    generate_documentation(analysis)
    create_instructions(analysis)
    recreate_code(analysis, local_path)

    print("Agentic Code Recreation System process completed successfully.")

if __name__ == "__main__":
    main()