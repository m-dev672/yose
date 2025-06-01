from gensim.models import KeyedVectors
import MeCab

from yose import Word2Vec, Screening, MPJoin, SubspaceWRD, Clustering

def tokenizer(sentence):
    mt = MeCab.Tagger('-Owakati')
    return mt.parse(sentence).strip()

def main():
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = f.readlines()

    tokenized_s = list(map(tokenizer, sentences))
    kv = KeyedVectors.load_word2vec_format('jawiki.word_vectors.200d.txt')
    word2vec = Word2Vec(kv)
    s_vecs = word2vec.run(tokenized_s)

    screening = Screening()
    sentences, tokenized_s, s_vecs = screening.run(sentences, tokenized_s, s_vecs)

    mpjoin = MPJoin(min_cos_sim=0.70)
    combinations = mpjoin.run(tokenized_s)

    subspace_wrd = SubspaceWRD()
    combinations, distances = subspace_wrd.run(s_vecs, combinations)

    clustering = Clustering()
    clustering.run(sentences, combinations, distances)

if __name__ == '__main__':
    main()