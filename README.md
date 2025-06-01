# yose (Yet anOther Sentence clustEring)

## What is "yose".
`yose` is a Python module for natural language processing and clustering of sentences.
It supports preprocessing, distance calculation between sentences, and clustering.
The name comes from the Japanese word `寄せる`, meaning "to come near".

## Feature
- 🚀 **Fast Extraction**：SimHash + MPJoin for fast pair extraction
- 🧠 **Highly accurate**：Calculations based on Word Rotator's Distance
- 🧵 **GPU parallel processing**: GPU parallel processing by Subspace Optimal Transport
- 📊 **Spectral Clustering**: Natural clustering based on distance
- 📁 **HTML Output**：Beautiful HTML output of cluster results
- 🇯🇵 **日本語対応**：MeCab を使ったトークナイズに対応

## Installation
Install yose from PyPI as:
```
pip install yose
```

## Example
```
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
```

## Attention
CPU processing has not been tested. Unexpected bottlenecks can occur.

## Citation

- Yokoi, Sho, et al. "Word Rotator's Distance." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.  
  [https://doi.org/10.18653/v1/2020.emnlp-main.236](https://doi.org/10.18653/v1/2020.emnlp-main.236)

- 黄健明, 笠井裕之. "異なるサイズの特徴量を対象としたGPU並列化可能な部分空間最適輸送手法の検討." 2023年度人工知能学会全国大会論文集.  
  [https://doi.org/10.11517/pjsai.JSAI2023.0_2T4GS504](https://doi.org/10.11517/pjsai.JSAI2023.0_2T4GS504)
