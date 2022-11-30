import pandas as pd
import spacy


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    nlp = spacy.load("ru_core_news_sm")
    for context in range(df.shape[0]):
      df['word'].iloc[context] = nlp(df['word'].iloc[context])[0].lemma_ + '_' + nlp(df['word'].iloc[context])[0].pos_
      df['context'].iloc[context] = ' '.join([token.lemma_ + '_' + token.pos_ for token in nlp(df['context'].iloc[context]) if len(token) > 1])
    return df
