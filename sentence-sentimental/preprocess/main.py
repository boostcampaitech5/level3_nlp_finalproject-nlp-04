from preprocessing import *
import pandas as pd

if __name__ == '__main__' :
    df = pd.read_csv("/opt/ml/finance_sentiment_corpus/뉴스_corpus.csv")
    
    df = df[['title', 'date', 'content']]
    
    # corpus
    df = preprocess_dataframe(df)
    df.to_csv("/opt/ml/finance_sentiment_corpus/preprocessed_data.csv")
    
    # sentence
    df_sentence = preprocess_dataframe_to_sentence(df)
    df_sentence.to_csv("/opt/ml/finance_sentiment_corpus/preprocessed_sentence_data.csv")