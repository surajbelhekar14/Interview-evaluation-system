import re
import spacy
from docx import Document
from PyPDF2 import PdfReader
import os
import json

# Load SpaCy model for name entity recognition
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a DOCX file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
    return text

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to determine the file type and extract text
def extract_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == ".docx":
        return extract_text_from_docx(file_path)
    elif file_extension.lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .docx or .pdf file.")

# Function to extract the name
def extract_name(text):
    # First, try using SpaCy's PERSON entity
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    
    # Fallback 1: Look for lines starting with "Name:"
    name_pattern = re.search(r"(?i)Name:\s*(.+)", text)
    if name_pattern:
        return name_pattern.group(1).strip()
    
    # Fallback 2: Assume the first non-empty line contains the name
    lines = text.split("\n")
    for line in lines:
        if line.strip():  # Non-empty line
            return line.strip()
    
    return "Name not found"

# Function to extract skills based on keywords and skills section
def extract_skills(text):
    skills_keywords = [
        "Python", "Java", "C++", "HTML", "CSS", "JavaScript", "SQL", "MySQL", "MongoDB",
        "Data Analysis", "Machine Learning", "Deep Learning", "AI", "NLP", "Django", "Flask",
    ]
    skills_section = re.search(r"(Skills|Technologies|Expertise)([\s\S]*?)(\n\n|\Z)", text, re.IGNORECASE)
    if skills_section:
        section_text = skills_section.group(2)
        skills_found = [skill for skill in skills_keywords if skill.lower() in section_text.lower()]
    else:
        skills_found = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return skills_found

# Function to extract projects based on keywords and project section
def extract_projects(text):
    projects_section = re.search(r"(Projects|Professional Experience|Work Experience)([\s\S]*?)(\n\n|\Z)", text, re.IGNORECASE)
    if projects_section:
        projects_text = projects_section.group(2).strip()
        projects_list = [line.strip() for line in projects_text.split("\n") if line.strip()]
        return "\n".join(projects_list)
    return "No projects found"

# Main function to parse resume and save output in JSON
def parse_resume(file_path, output_path="resume_output.json"):
    try:
        text = extract_text(file_path)
        name = extract_name(text)
        skills = extract_skills(text)
        projects = extract_projects(text)

        # Store results in a dictionary
        result = {
            "Name": name,
            "Skills": skills,
            "Projects": projects
        }

        # Save the results to a JSON file with UTF-8 encoding and ensure_ascii=False
        with open(output_path, "w", encoding='utf-8') as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

        print(f"Resume analysis saved to {output_path}")
        return result

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    # Path to the resume file (.pdf or .docx)
if __name__ == "__main__":
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    # ... rest of your if __name__ == "__main__": code ...
    uploads_folder = "uploads"

    try:
        files_in_uploads = os.listdir(uploads_folder)
        if files_in_uploads:
            # Assuming the first file in the list is the one you want to process
            first_file_name = files_in_uploads[-1]
            file_path = os.path.join(uploads_folder, first_file_name)

            if os.path.exists(file_path):
                result = parse_resume(file_path)
                print("Resume Analysis:")
                if result:
                    print(f"Name: {result['Name']}")
                    print(f"Skills: {', '.join(result['Skills'])}")
                    print(f"Projects: {result['Projects']}")
            else:
                print(f"Error: File not found at {file_path}.")
        else:
            print(f"Error: No files found in the '{uploads_folder}' directory.")

    except FileNotFoundError:
        print(f"Error: The directory '{uploads_folder}' was not found. Make sure it exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print("Resume Analysis:")
    print(f"Name: {result['Name']}")
    print(f"Skills: {', '.join(result['Skills'])}")
    print(f"Projects: {result['Projects']}")
