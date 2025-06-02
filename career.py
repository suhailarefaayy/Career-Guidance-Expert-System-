import streamlit as st
import pandas as pd
import numpy as np
import ast
import difflib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from experta import KnowledgeEngine, Fact, DefFacts, Rule, MATCH
import spacy
import collections
import collections.abc

# Compatibility fix for deprecated Mapping
collections.Mapping = collections.abc.Mapping

# Initialize NLP and NLTK
@st.cache_resource
def load_nlp_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return spacy.load("en_core_web_sm")

nlp = load_nlp_resources()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# File paths
RAW_DATA_PATH = "Career Guidance Expert System.csv"
CLEANED_DATA_PATH = "cleaned_dataset.csv"

# Load and preprocess dataset
@st.cache_data
def load_and_prepare_data():

    try:
        data = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File {RAW_DATA_PATH} not found.")
        
        data = None
        data['soft_skill'] = data['soft_skill'].replace("[]", np.nan)
        data['hard_skill'] = data['hard_skill'].replace("[]", np.nan)

#just filling the  nulls with mode values
    soft_modes = data.groupby('candidate_field')['soft_skill'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    hard_modes = data.groupby('candidate_field')['hard_skill'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()

    def fill_skills(row):
        if pd.isnull(row['soft_skill']):
            row['soft_skill'] = soft_modes.get(row['candidate_field'], row['soft_skill'])
        if pd.isnull(row['hard_skill']):
            row['hard_skill'] = hard_modes.get(row['candidate_field'], row['hard_skill'])
        return row

    data = data.apply(fill_skills, axis=1)
#-----------------------------------------------------------------
    # Convert string to list 
    def parse_skill_list(skill_str):
        if pd.isna(skill_str) or skill_str == "[]":
            return []
        try:
            # Handle various formats in the data
            cleaned = skill_str.replace("'", '"').replace('""', '"')
            if not cleaned.startswith('['):
                cleaned = f"[{cleaned}]"
            skills = ast.literal_eval(cleaned)
            return list(set([s.strip().lower() for s in skills if s.strip()]))
        except (SyntaxError, ValueError) as e:
            st.error(f"Error parsing skill list: {skill_str} - {e}")
            return []

    data['hard_skill'] = data['hard_skill'].apply(parse_skill_list)
    data['soft_skill'] = data['soft_skill'].apply(parse_skill_list)
    data['candidate_field'] = data['candidate_field'].str.strip().str.lower()


    # Prioritize entries with label=1 (suitable matches)
    data_labeled = data.copy()
    data_labeled['priority'] = data_labeled['label'].apply(lambda x: 1 if x == 1 else 0)
    data_labeled = data_labeled.sort_values('priority', ascending=False)
    
    data.to_csv(CLEANED_DATA_PATH, index=False)
    return data_labeled

# Load cleaned data
df = load_and_prepare_data()

# Build field skill dictionary with weights based on frequency
field_skills = {}
for field, group in df.groupby('candidate_field'):
    # Count occurrences of each hard skill
    hard_skills = sum(group['hard_skill'].tolist(), [])
    hard_count = pd.Series(hard_skills).value_counts().to_dict()
    
    # Count occurrences of each soft skill
    soft_skills = sum(group['soft_skill'].tolist(), [])
    soft_count = pd.Series(soft_skills).value_counts().to_dict()
    
    field_skills[field] = {
        'hard': hard_count,
        'soft': soft_count
    }

# Get all unique skills set alshan tekrar 
all_hard_skills = set()
all_soft_skills = set()
for field_data in field_skills.values():
    all_hard_skills.update(field_data['hard'].keys())
    all_soft_skills.update(field_data['soft'].keys())

all_known = sorted(all_hard_skills | all_soft_skills)

# Enhanced NLP function
def nlp_input(text):
    doc = nlp(text.lower())
    extracted_terms = []

    # Extract noun chunks (multi-word terms)
    for chunk in doc.noun_chunks:
        chunk_text = ' '.join([token.lemma_.lower() for token in chunk
                              if token.text.lower() not in stop_words
                              and token.text not in string.punctuation
                              and not token.is_space])
        if chunk_text and len(chunk_text) > 1:
            extracted_terms.append(chunk_text)

    # Extract individual tokens not part of chunks
    for token in doc:
        if (token.text.lower() not in stop_words and 
            token.text not in string.punctuation and 
            not token.is_space and
            len(token.text) > 1):
            
            in_chunk = any(chunk.start <= token.i < chunk.end for chunk in doc.noun_chunks)
            if not in_chunk:
                extracted_terms.append(token.lemma_.lower())

    return list(set(extracted_terms))

# Match extracted terms to skills and classify them
def match_and_classify_skills(terms):
    matched_hard = []
    matched_soft = []
    
    for term in terms:
        # Try direct matching first
        if term in all_hard_skills:
            matched_hard.append(term)
            continue
            
        if term in all_soft_skills:
            matched_soft.append(term)
            continue
        
        # Try fuzzy matching if direct match fails
        hard_match = None
        soft_match = None
        hard_score = 0
        soft_score = 0
        
        # Check against hard skills
        for skill in all_hard_skills:
            score = difflib.SequenceMatcher(None, term, skill).ratio()
            if score > hard_score and score >= 0.7:  # Higher threshold for better accuracy
                hard_score = score
                hard_match = skill
                
        # Check against soft skills
        for skill in all_soft_skills:
            score = difflib.SequenceMatcher(None, term, skill).ratio()
            if score > soft_score and score >= 0.7:  # Higher threshold for better accuracy
                soft_score = score
                soft_match = skill
        
        # Add the best match
        if hard_match and hard_score > soft_score:
            matched_hard.append(hard_match)
        elif soft_match:
            matched_soft.append(soft_match)
    
    return list(set(matched_hard)), list(set(matched_soft))

class CareerInput(Fact): 
    pass

class CareerExpert(KnowledgeEngine):
    @DefFacts()
    def _startup(self):
        yield Fact(action="recommend")

    @Rule(Fact(action="recommend"), 
          CareerInput(hard_skills=MATCH.hard_skills, 
                     soft_skills=MATCH.soft_skills))
    def _recommend(self, hard_skills, soft_skills):
        # Calculate weighted scores for each field
        field_scores = {}
        confidence_scores = {}
        
        total_skills = len(hard_skills) + len(soft_skills)
        if total_skills == 0:
            st.warning(" No clear skills detected in your input")
            return
            
        
        for field, skills in field_skills.items():
            # Hard skills matching
            hard_match_score = 0
            for skill in hard_skills:
                # Get the frequency weight of this skill in this field
                weight = skills['hard'].get(skill, 0)
                hard_match_score += weight if weight > 0 else 0
            
            # Soft skills matching
            soft_match_score = 0
            for skill in soft_skills:
                # Get the frequency weight of this skill in this field
                weight = skills['soft'].get(skill, 0)
                soft_match_score += weight if weight > 0 else 0
            
            # Calculate total score with more weight on hard skills
            total_score = (hard_match_score * 1.5) + soft_match_score
            
            # Normalize by total skill count
            field_size = sum(skills['hard'].values()) + sum(skills['soft'].values())
            if field_size > 0:
                normalized_score = total_score / field_size
                field_scores[field] = normalized_score
                
                # Calculate confidence based on how many skills matched
                hard_skills_matched = sum(1 for skill in hard_skills if skill in skills['hard'])
                soft_skills_matched = sum(1 for skill in soft_skills if skill in skills['soft'])
                total_matched = hard_skills_matched + soft_skills_matched
                
                if total_matched > 0:
                    # Calculate confidence percentage
                    confidence = (total_matched / total_skills) * 100
                    confidence_scores[field] = min(confidence, 100)  # Cap at 100%
        
        if field_scores:
            # Get the best match
            best_field = max(field_scores, key=field_scores.get)
            confidence = confidence_scores.get(best_field, 0)
            
            # Display with confidence
            if confidence >= 70:
                st.success(f" A suitable candidate field for you would be **{best_field.title()}** (Confidence: {confidence:.1f}%)")
            elif confidence >= 40:
                st.info(f" A likely candidate field for you could be **{best_field.title()}** (Confidence: {confidence:.1f}%)")
            else:
                st.info(f" You might consider exploring **{best_field.title()}** as a career option (Confidence: {confidence:.1f}%)")
            
            # Show matched skills
            st.write("### Skills matched to this field:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Technical Skills:**")
                matched_hard_in_field = [s for s in hard_skills if s in field_skills[best_field]['hard']]
                if matched_hard_in_field:
                    for skill in matched_hard_in_field:
                        st.write(f"- {skill}")
                else:
                    st.write("- No specific technical skills matched")
                    
            with col2:
                st.write("**Soft Skills:**")
                matched_soft_in_field = [s for s in soft_skills if s in field_skills[best_field]['soft']]
                if matched_soft_in_field:
                    for skill in matched_soft_in_field:
                        st.write(f"- {skill}")
                else:
                    st.write("- No specific soft skills matched")
                    
      
        else:
            st.warning(" No clear career match found based on the skills provided.")

# Streamlit GUI PART
st.title(" Career Guidance Expert System")
st.write("Describe your skills or experience")

user_input = st.text_area(" Describe your skills and experience:", 
                          height=150, 
                          help="For example: 'I have experience in nursing, registration, and service. I am also good at written communication.'")

if st.button(" Get Career Recommendation"):
    if user_input.strip():
        
        extracted_terms = nlp_input(user_input)
        hard_skills, soft_skills = match_and_classify_skills(extracted_terms)
        
        
        st.write("### Extracted Skills")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Technical Skills:**")
            if hard_skills:
                for skill in hard_skills:
                    st.write(f"- {skill}")
            else:
                st.write("- No specific technical skills detected")
                
        with col2:
            st.write("**Soft Skills:**")
            if soft_skills:
                for skill in soft_skills:
                    st.write(f"- {skill}")
            else:
                st.write("- No specific soft skills detected")
        
        
        engine = CareerExpert()
        engine.reset()
        engine.declare(CareerInput(hard_skills=hard_skills, soft_skills=soft_skills))
        engine.run()
    else:
        st.error(" Please enter some text.")