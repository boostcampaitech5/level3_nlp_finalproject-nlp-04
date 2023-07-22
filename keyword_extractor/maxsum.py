import itertools
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def max_sum_sim(
		doc_embedding,
		word_embedding,
		words,
		nr_candidates: int,
		top_k: int = 5,
		):
	if nr_candidates < top_k:
		raise Exception("고려할 후보 군의 갯수가 Top-K보다 작습니다.")
	elif top_k > len(words):
		return []

	dis = cosine_similarity(doc_embedding, word_embedding)
	dis_words = cosine_similarity(word_embedding)

	words_idx = list(dis.argsort()[0][-nr_candidates:])
	words_vals = [words[idx] for idx in words_idx]
	candidates = dis_words[np.ix_(words_idx, words_idx)]

	min_sim = 100_000
	candidate = None

	for combination in itertools.combinations(range(len(words_idx)), top_k):
		sim = sum([candidates[i][j] for i in combination for j in combination if i != j])

		if sim < min_sim:
			candidate = combination
			min_sim = sim

	return [(words_vals[idx], round(float(dis[0][words_idx[idx]]), 3)) for idx in candidate]