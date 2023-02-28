import re
import pandas as pd
import spacy
from spacy.cli import download

class senTM:

    def __init__(self):
        self.lexico = pd.read_csv(
            "https://raw.githubusercontent.com/dsldk/danish-sentiment-lexicon/main/2_headword_headword_polarity.csv", 
            header=None,
            names=['headword','homographs','pos','id','polarity','wordforms']
        )

        self.pos_dict = {
            "adj.": "ADJ",
            "sb.": "NOUN",
            "sb. pl.": "NOUN", 
            "vb.": "VERB",
            "adv.": "ADV",
            "udråbsord": "INTJ",
            "sidsteled": "X",
            "egennavn": "PROPN", 
            "fork.": "NOUN", 
            "førsteled": "X",
            "adj. førsteled": "X",
            "konj.": "CCONJ",
            "lydord": "INTJ",
            "pron.": "PRON",
            "præfiks": "X",
            "NA": "X"
        }

        self.lexico['pos'] = self.lexico['pos'].fillna("NA").apply(lambda x: self.pos_dict[x])
        self.lexico['all_words'] = self.lexico['headword'].apply(lambda x: [x]) + self.lexico['wordforms'].fillna('').apply(lambda x: x.split(';'))
        self.lexico['all_words'] = self.lexico['all_words'].apply(lambda x: list(set(x)))
        self.lexico = self.lexico.explode('all_words') #[['all_words', 'pos', 'polarity']]


        try:
            self.spacy_pipeline = spacy.load("da_core_news_sm")
        except OSError:
            print('Downloading language model for the spaCy POS tagger\n'
                "(don't worry, this will only happen once)")
            download('en')
            self.spacy_pipeline = spacy.load("da_core_news_sm")
        

    def tokenizer(self, text: str) -> list:
        # return [token.lower() for token in re.split('\W', text) if token != '']
        doc = self.spacy_pipeline(text)
        return [token for token in doc]
 
    def mean(self, list_of_homograph_scores: list) -> float:
        return sum(list_of_homograph_scores) / len(list_of_homograph_scores)
     
    def find_matches(self, text) -> list:
        tokens = self.tokenizer(text)
        matched_tokens = []
        for token in tokens:
            matches = self.lexico[
                self.lexico['all_words'] == token.text
            ]

            if matches.shape[0] == 0:
                continue
            elif matches[matches['pos'] == token.pos_].shape[0] == 1:
                matched_tokens.append(
                    (token.text, matches[matches['pos'] == token.pos_]['polarity'].iloc[0])
                )
            elif matches[(matches['pos'] == token.pos_) & (matches['headword'] == token.text)].shape[0] == 1:
                matched_tokens.append(
                    (token.text, matches[(matches['pos'] == token.pos_) & (matches['headword'] == token.text)]['polarity'].iloc[0])
                )
            elif matches[matches['pos'] == token.pos_].shape[0] > 1:
                matched_tokens.append(
                    (token.text, self.mean(matches['polarity'].tolist()))
                )

        return matched_tokens
           
    def find_scores(self, text: str) -> list:
        return [match[1] for match in self.find_matches(text)]
    
    def score(self, text):
        return float(sum(self.find_scores(text)))
    
    def classify(self, text) -> str:
        score = self.score(text)
        if score > 1:
            return 'positiv'
        elif score < 1 and score > -1:
            return 'neutral'
        else:
            return 'negativ'
        