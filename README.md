
# senTM

Pronounced "sen-T-M". 

senTM is a lexical approach to sentiment analysis in Danish. The approach is inspired by the [afinn](https://github.com/fnielsen/afinn) package, but using the [Danish Sentiment Lexicon version 0.2 (2022-12-20)](https://github.com/dsldk/danish-sentiment-lexicon). 

The approach use the part-of-speech (POS) model from Spacy and then matches any tokens from the Danish Sentiment Lexicon. 

License: Same as [Danish Sentiment Lexicon version 0.2 (2022-12-20)](https://github.com/dsldk/danish-sentiment-lexicon): CC-BY-SA 4.0 International https://creativecommons.org/licenses/by-sa/4.0/

## Installation

Install from pip:
```
pip install sentm
```


## Quickstart

First initialize model. 
```
from sentm.sentm import senTM

sentm_model = senTM()
```

You can both get sentiment score:
```
sentm_model.score('Du er en kæmpe idiot!')
```

You can also use it as a classifier. 
Here, the labels are determined by:
* Score larger than 1: "positiv"
* Score between -1 and 1: "neutral"
* Score lower than -1: "negativ"
```
sentm_model.classify('Du er en kæmpe idiot!')
```