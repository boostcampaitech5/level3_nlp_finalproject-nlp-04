import numpy as np

from typing import List, Union, Tuple

from konlpy.tag import Mecab, Okt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keyword_extractor.mmr import mmr
from keyword_extractor.maxsum import max_sum_sim

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
						   top_k,
						   diversity)

			keywords = self.post_processing(keywords, tag_type, stop_words)

			all_keywords.append(keywords)

		return all_keywords


	def post_processing(self,
						keywords,
						tag_type:str = None,
						stop_words:Union[str, List[str]] = [],
						) -> List[Tuple[str, float]]:
		"""Post-processing for extracted keywords

		추출된 키워드의 오른쪽 경계의 품사를 확인하고, 명사가 아니면 차례로 제거

		Args:
			keywords:
			 	Keywords to post-process
			tag_type:
			 	Tagger type for post-processing
			stop_words:
			 	Stop words for post-processing

		Returns:
			List of keywords and scores:
				[("keyword1", score1), ("keyword2", score2), ...]
		"""
		assert tag_type is not None, "tag_type을 입력해주세요."

		processed_keywords = []

		if tag_type == "mecab":
			tagger = Mecab()
		elif tag_type == "okt":
			tagger = Okt()

		for word, score in keywords:
			ap = tagger.pos(word)

			while len(ap) != 0 and "N" != ap[-1][1][0]: # 맨 우측에서부터 명사가 아닌 것을 제거
				ap.pop()

			keyword = "".join([a[0] for a in ap])

			if keyword and (keyword not in stop_words): # 키워드가 남아있으면, 추가
				processed_keywords.append((keyword, score))

		return processed_keywords