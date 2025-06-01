from gensim.models import KeyedVectors
import MeCab

from yose import Word2Vec, Screening, MPJoin, SubspaceWRD, Clustering

def tokenizer(sentence):
    mt = MeCab.Tagger('-Owakati')
    return mt.parse(sentence).strip()

def main():
    # Read sentences from a file (one sentence per line)
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = f.readlines()

    # Tokenize each sentence
    tokenized_s = list(map(tokenizer, sentences))

    # Load a pre-trained Word2Vec model
    kv = KeyedVectors.load_word2vec_format('jawiki.word_vectors.200d.txt')
    word2vec = Word2Vec(kv)

    # Generate sentence vectors from tokenized sentences
    s_vecs = word2vec.run(tokenized_s)

    # Filter out invalid sentences:
    # - Removes sentences that is empty
    # - Also excludes sentences whose length is more than two standard deviations above or less the mean
    screening = Screening()
    sentences, tokenized_s, s_vecs = screening.run(sentences, tokenized_s, s_vecs)

    # Find sentence pairs with high cosine similarity (>= 0.70)
    mpjoin = MPJoin(min_cos_sim=0.70)
    combinations = mpjoin.run(tokenized_s)

    # Calculate distances between sentence pairs using Subspace Word Rotational Distance
    subspace_wrd = SubspaceWRD()
    combinations, distances = subspace_wrd.run(s_vecs, combinations)

    # Perform clustering based on caliculated distances
    # - The results are saved to output.html by default.
    clustering = Clustering()
    clustering.run(sentences, combinations, distances)

if __name__ == '__main__':
    main()
