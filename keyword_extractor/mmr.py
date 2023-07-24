import numpy as np

from typing import List, Tuple

from sklearn.metrics.pairwise import cosine_similarity

def mmr(
		doc_embedding,
		word_embedding,
		words,
		top_k: int = 5,
		diversity: float = 0.7,
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

	keywords_idx = [np.argmax(word_doc_sim)]
	candidate_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

	for _ in range(min(top_k-1, len(words)-1)):
		candidate_sim = word_doc_sim[candidate_idx, :]
		target_sim = np.max(word_sim[candidate_idx][:, keywords_idx], axis=1)

		mmr = (1-diversity) * candidate_sim - diversity * target_sim.reshape(-1, 1)
		mmr_idx = candidate_idx[np.argmax(mmr)]

		keywords_idx.append(mmr_idx)
		candidate_idx.remove(mmr_idx)

	keywords = [(words[idx], round(float(word_doc_sim.reshape(1, -1)[0][idx]), 3)) for idx in keywords_idx]
	keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

	return keywords