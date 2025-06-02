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
collections.Mapping = collections.abc.Mapping
data=pd.read_csv("Career Guidance Expert System.csv")
data['soft_skill'] = data['soft_skill'].replace("[]", np.nan)
data['hard_skill'] = data['hard_skill'].replace("[]", np.nan)

#getting modes of skills at each unique candidate field
soft_skill_modes = (
    data.groupby('candidate_field')['soft_skill']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .to_dict()
)
hard_skill_modes = (
    data.groupby('candidate_field')['hard_skill']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .to_dict()
)

#filling nulls
def fill_skills(row):
    if pd.isnull(row['soft_skill']):
        row['soft_skill'] = soft_skill_modes.get(row['candidate_field'], row['soft_skill'])
    if pd.isnull(row['hard_skill']):
        row['hard_skill'] = hard_skill_modes.get(row['candidate_field'], row['hard_skill'])
    return row



#apply to df
data = data.apply(fill_skills,axis=1)
data['hard_skill'] = data['hard_skill'].apply(lambda x: list(set(x)))
data['soft_skill'] = data['soft_skill'].apply(lambda x: list(set(x)))
data.to_csv("cleaned_dataset.csv", index=False)
# #convert to actual lists
data['hard_skill'] = data['hard_skill'].apply(lambda x: [s.strip().lower() for s in ast.literal_eval(x)])
data['soft_skill'] = data['soft_skill'].apply(lambda x: [s.strip().lower() for s in ast.literal_eval(x)])
#convert to lower and remove whitespaces
data['candidate_field'] = data['candidate_field'].str.strip().str.lower()
df = pd.read_csv("cleaned_dataset.csv")
# Initialize NLP
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and preprocess dataset
df = pd.read_csv("cleaned_dataset.csv")
df['hard_skill'] = df['hard_skill'].apply(eval)
df['soft_skill'] = df['soft_skill'].apply(eval)

# Build field_skills dictionary
field_skills = {}
for field, group in df.groupby('candidate_field'):
    hard = set(sum(group['hard_skill'].tolist(), []))
    soft = set(sum(group['soft_skill'].tolist(), []))
    field_skills[field] = {'hard': hard, 'soft': soft}

all_known = sorted(set().union(*((req['hard'] | req['soft']) for req in field_skills.values())))

# NLP function for skill extraction
def nlp_input(text):
    doc = nlp(text)
    cleaned = []

    for chunk in doc.noun_chunks:
        chunk_text = ' '.join([token.lemma_.lower() for token in chunk
                              if token.text.lower() not in stop_words
                              and token.text not in string.punctuation
                              and not token.is_space])
        if chunk_text and len(chunk_text) > 1:
            cleaned.append(chunk_text)

    for token in doc:
        if token.text.lower() not in stop_words and token.text not in string.punctuation and not token.is_space:
            exist = any(chunk.start <= token.i < chunk.end for chunk in doc.noun_chunks)
            if not exist:
                cleaned.append(token.lemma_.lower())

    return list(set(cleaned))

# Experta setup
class CareerInput(Fact):
    pass

class CareerExpert(KnowledgeEngine):

    @DefFacts()
    def _startup(self):
        yield Fact(action="recommend")

    @Rule(Fact(action="recommend"), CareerInput(text=MATCH.text))
    def _recommend(self, text):
        tokens = nlp_input(text)
        matched = []

        for tok in tokens:
            best, score = None, 0.0
            for s in all_known:
                r = difflib.SequenceMatcher(None, tok, s).ratio()
                if r > score:
                    best, score = s, r
            if best and score >= 0.6:
                matched.append(best)

        counts = {}
        for field, req in field_skills.items():
            counts[field] = sum(s in req['hard'] for s in matched) + \
                            sum(s in req['soft'] for s in matched)

        if any(counts.values()):
            best_field = max(counts, key=counts.get)
            st.success(f"✅A suitable candidate field for you would be **{best_field.title()}**")
        else:
            st.warning("⚠️ No clear career match found.")

# --- Streamlit GUI ---
st.title("💼 Career Guidance expert system")
st.write("Enter your experience or skills and get a suggested career field.")

user_input = st.text_area("🧠 Describe your skills and experience:")

if st.button("🔍 Get Career Recommendation"):
    if user_input.strip():
        engine = CareerExpert()
        engine.reset()
        engine.declare(CareerInput(text=user_input))
        engine.run()
    else:
        st.error("Please enter some input text.")
