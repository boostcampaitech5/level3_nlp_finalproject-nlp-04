import numpy as np
import math

from typing import List, Union, Tuple

from konlpy.tag import Mecab, Okt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keyword_extractor.mmr import mmr
from keyword_extractor.maxsum import max_sum_sim

import torch

class KeyBert:
	"""Keyword Extractor using Sentence-BERT


	Attributes:
		model: Sentence-BERT model
	"""
	def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> None:
		"""Initialize Sentence-BERT Model Object

		Args:
			model_name: Sentence-BERT model name in HuggingFace
		"""
		self.model = SentenceTransformer(model_name)

	def extract_keywords(self,
						 docs: Union[str, List[str]],
						 titles: Union[str, List[str]] = None,
						 keyphrase_ngram_range: Tuple[int, int]=(1, 1),
						 stop_words:Union[str, List[str]] = [],
						 top_k:int = 5,
						 diversity:float = 0.7,
						 min_df:int = 1,
						 candidate_frac:float = 0.3,
						 vectorizer_type:str = "tfidf",
						 tag_type:str = "mecab",
						 ) -> List[Tuple[str, float]]:
		"""Extract keywords from documents

		Vectorizer를 이용해서, 문서에서 후보 키워드를 추출
		추출된 후보 키워드를 Sentence-BERT를 이용해서 문서와의 유사도(MMR)를 계산

		Args:
			titles:
				Titles to extract keywords
			docs:
			 	Documents to extract keywords
			keyphrase_ngram_range:
			 	N-gram range for keyphrase
			stop_words:
			 	Stop words for vectorizer
			top_k:
			 	Number of keywords to extract
			diversity:
			 	Diversity for MMR - (0 ~ 1)
			min_df:
			 	Minimum document frequency for vectorizer
			candidate_frac:
			 	Fraction of candidates to extract keywords - (0 ~ 1)
			vectorizer_type:
			 	Vectorizer type for extracting keywords - ["count", "tfidf"]
			tag_type:
			 	Tagger type for extracting keywords - ["mecab", "okt"]

		Returns:
			List of keywords and scores:
				[("keyword1", score1), ("keyword2", score2), ...]

		"""

		if isinstance(docs, str):
			if docs:
				docs = [docs]
			else:
				return []

		if isinstance(titles, str):
			if titles:
				titles = [titles]
			else:
				return []

		if isinstance(stop_words, str):
			if stop_words:
				stop_words = [stop_words]
			else:
				stop_words = []

		# Count Vectorizer
		if vectorizer_type == "count":
			vectorizer = CountVectorizer(ngram_range=keyphrase_ngram_range, min_df=min_df, stop_words=stop_words).fit(docs)
		# TF-IDF Vectorizer
		elif vectorizer_type == "tfidf":
			vectorizer = TfidfVectorizer(ngram_range=keyphrase_ngram_range, min_df=min_df, stop_words=stop_words).fit(docs)


		words = vectorizer.get_feature_names_out()  # Vocab이 뽑힘
		df = vectorizer.transform(docs)

		doc_embs = self.model.encode(docs)
		word_embs = self.model.encode(words)

		all_keywords = []

		for i, _ in enumerate(docs):
			df_value = df[i].data
			top_frac_candidate_idx = df[i].nonzero()[1][np.argsort(df_value)[::-1][:int(len(df_value) * candidate_frac)]]
			# candidate_idx = df[i].nonzero()[1]
			candidate_words = [words[idx] for idx in top_frac_candidate_idx]
			candidate_emb = word_embs[top_frac_candidate_idx]

			doc_emb = doc_embs[i].reshape(1, -1)

			keywords = mmr(doc_emb,
						   candidate_emb,
						   candidate_words,
						   stop_words,
						   top_k,
						   diversity,
						   tag_type)

			# keywords = self.post_processing(keywords, tag_type, stop_words)

			all_keywords.append(keywords)

		return all_keywords

class KeyExtract:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.okt = Okt()

    def get_words(self, text, stopwords = []):
        words = []
        for word in self.okt.phrases(text):
            if len(word.split()) == 1:
                if stopwords and (word in stopwords):
                    continue
                words.append(word)
        return words
        
    def extract_keywords(self,
                         docs: Union[str, List[str]],
                         titles: Union[str, List[str]] = None,
                         keyphrase_ngram_range=(1, 1),
                         stop_words=[],
                         top_k:int=5,
                         diversity=0.7,
                         min_df=1,
                         candidate_frac=0.3,
                         vectorizer_type="tfidf",
                         tag_type="mecab",
                         max_count:int = 5,
                         ):

        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []
                
        if isinstance(titles, str):
            if titles:
                titles = [titles]
            else:
                return []
                
        if isinstance(stop_words, str):
            if stop_words:
                stop_words = [stop_words]
            else:
                stop_words = []
                
        all_keywords = []
        for i in range(len(docs)):
            if titles:
                doc = titles[i] + docs[i]
            else:
                doc = docs[i]
            #키워드 후보 가져오기 가져오기
            words = self.get_words(doc, stop_words)
            
            #코사인 유사도
            doc_emb = self.model.encode(doc, device=self.device)
            word_emb = self.model.encode(words, device=self.device)
            
            text_word_cos = cosine_similarity([doc_emb], word_emb).flatten()
            
            #빈도수 점수
            word_counts = {word: doc.count(word) for word in words}
            frequency_score = []
            for count in word_counts.values():
                n = count if count < max_count else max_count
                frequency_score.append(math.log(n + 1, 100))
            frequency_score = np.array(frequency_score)

            #score 계산
            keywords = []
            score = text_word_cos * frequency_score
            for i in np.argsort(-score):
                keywords.append((words[i], score[i]))
            if top_k:
                keywords = [(word, score) for word, score in keywords][:top_k]
            else:
                keywords = [(word, score) for word, score in keywords]
            all_keywords.append(keywords)
        return all_keywords
        