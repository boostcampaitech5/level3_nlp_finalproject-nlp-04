import numpy as np

from typing import List, Tuple, Union

from konlpy.tag import Mecab, Okt
from sklearn.metrics.pairwise import cosine_similarity

def mmr(
		doc_embedding,
		word_embedding,
		words,
		stop_words: Union[str, List[str]] = [],
		top_k: int = 5,
		diversity: float = 0.7,
		tag_type: str = "okt",
		) -> List[Tuple[str, float]]:
	"""Maximal Marginal Relevance

	Calculate MMR score for each word and extract top-k keywords

	Args:
		doc_embedding:
			Embedding of document
		word_embedding:
			Embedding of words
		words:
			Words to extract keywords
		stop_words:
			Stop words for vectorizer
		top_k:
			Number of keywords to extract
		diversity:
			Diversity for MMR - (0 ~ 1)

	Returns:
		Keywords and scores: List of tuples (word, score)
	"""

	assert 0 <= diversity <= 1, "Diversity should be between 0 and 1"

	word_doc_sim = cosine_similarity(word_embedding, doc_embedding)
	word_sim = cosine_similarity(word_embedding)

	keywords = []
	keywords_idx = [np.argmax(word_doc_sim)]
	candidate_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

	while len(keywords) < min(top_k, len(words)):
		candidate_sim = word_doc_sim[candidate_idx, :]
		target_sim = np.max(word_sim[candidate_idx][:, keywords_idx], axis=1)

		mmr = (1-diversity) * candidate_sim - diversity * target_sim.reshape(-1, 1)
		# 짧은 기사에서 candidate_idx가 모두 지워지는 경우 에러 처리
		try:
			mmr_idx = candidate_idx[np.argmax(mmr)]
		except ValueError:
			print("Raise ValueError: mmr_idx")
			break

		post_processed_keyword = post_processing(words[mmr_idx],
												 tag_type=tag_type)
		if post_processed_keyword != "" and post_processed_keyword not in stop_words and post_processed_keyword not in [k[0] for k in keywords]:
			keywords.append((post_processed_keyword, round(float(word_doc_sim.reshape(1, -1)[0][mmr_idx]), 3)))
			candidate_idx.remove(mmr_idx)
		else:
			candidate_idx.remove(mmr_idx)

	# keywords = [(words[idx], round(float(word_doc_sim.reshape(1, -1)[0][idx]), 3)) for idx in keywords_idx]
	keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

	return keywords

def post_processing(keyword: str,
					tag_type:str = None,
					) -> str:
	"""Post-processing for extracted keywords

	추출된 키워드의 오른쪽 경계의 품사를 확인하고, 명사가 아니면 차례로 제거

	Args:
		keyword:
			Keyword to post-process
		tag_type:
			Tagger type for post-processing

	Returns:
		processd keyword:
			"processed keyword"
	"""
	assert tag_type is not None, "tag_type을 입력해주세요."


	if tag_type == "mecab":
		tagger = Mecab()
	elif tag_type == "okt":
		tagger = Okt()

	ap = tagger.pos(keyword)

	while len(ap) != 0 and "N" != ap[-1][1][0]: # 맨 우측에서부터 명사가 아닌 것을 제거
		ap.pop()

	keyword = "".join([a[0] for a in ap])

	processed_keyword = keyword

	return processed_keyword