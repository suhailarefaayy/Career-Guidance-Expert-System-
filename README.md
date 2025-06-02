# Career-Guidance-Expert-System-

## DESC
Building a sys that recommends career based on user's skills, Extract the skills soft and hard from user's input and analyze them then sys will infere the job matches.

## Dataset Overview:
**Each entry in the dataset includes the following fields**:
- ***hard_skill***: A list of technical or domain-specific skills possessed by the candidate (e.g., 'data analysis', 'project management').
- ***soft_skill***: A list of interpersonal or cognitive skills (e.g., 'teamwork', 'problem solving').
- ***label***: A binary indicator (0 or 1) denoting whether the candidate is a suitable match for a particular job role.
- ***candidate_field***: The professional domain or industry of the candidate (e.g., 'healthcare & medical', 'marketing').

### Run Example
**Input**: "I have experience in nursing, registration, and service. I am also good at written communication."  
**Output**: A suitable candidate field for you would be healthcare & medical.

### WORKFLOW OF THE PROJ:
- *Input Processing*: The user provides a description of their skills, typically in free-text format.

- **Skill Extraction**: The system processes the input using NLP techniques, including tokenization, named entity recognition (NER), and keyword extraction, to identify relevant skills from the user input.

- **Matching**: The extracted skills are compared with the skills in the job role dataset to find the best matches.

- **Inference**: Using a rule-based engine, the system matches the user’s skills to job roles based on predefined mappings and relevance scores.

- **Output**: The system returns a list of recommended job roles along with a confidence score for each suggestion. This allows the user to make an informed decision about their potential career path.
