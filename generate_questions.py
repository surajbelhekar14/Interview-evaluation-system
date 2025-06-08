import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load the .env file to securely fetch the API key
env_path = r".env"  # Adjust to your file path if needed
load_dotenv(env_path)

# Fetch the Gemini API key from the environment variable
API_KEY = 'AIzaSyAoOR8GkFRqgxgSPXukGiItH6PP7Zi8RbE'

if not API_KEY:
    raise ValueError("API key not found! Make sure the .env file is set up correctly.")

# Configure the Gemini API with the key
genai.configure(api_key=API_KEY)

# Function to generate interview questions based on resume data
def generate_interview_questions(json_file_path='resume_output.json', output_file='generated_questions.txt'):
    try:
        with open(json_file_path, "r") as json_file:
            resume_data = json.load(json_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_file_path} was not found. Make sure it exists.")
    except json.JSONDecodeError:
        raise ValueError(f"The file {json_file_path} is not a valid JSON file.")

    # Create the prompt using the full JSON content
    prompt = f"""
        You are an AI interviewer for a technical interview. Based on the following resume data, generate:
        1. 4 basic technical interview questions.
        2. 4 advanced technical interview questions.
        3. 2 HR interview questions related to the candidate's resume.
        while giveing questions dont assign a serial no to that questions.
        extract only name of candidate and store it in first line of document directly as it is without any modification.  
        Resume Data:
        {json.dumps(resume_data, indent=4)}
        """

    # Create the model instance (use the correct model name)
    model = genai.GenerativeModel("gemini-1.5-flash")  # Or a suitable Gemini model

    # Generate content using the model
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        raise ValueError(f"An error occurred while interacting with the Gemini API: {e}")

    # Extract the generated questions from the response
    questions = response.text

    # Save the generated questions to a file
    try:
        with open(output_file, "w") as file:
            file.write(questions)
        print(f"Questions saved to {output_file}")
        return questions
    except IOError:
        raise IOError("An error occurred while saving the generated questions to a file.")

if __name__ == "__main__":
    try:
        questions = generate_interview_questions()
        print("\nGenerated Questions:\n")
        print(questions)
    except Exception as e:
        print(f"Error: {e}")