import torch
# from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx.algorithms as A
# import gensim
# from gensim.models import KeyedVectors
def get_bow(raw_texts):
    from sklearn.feature_extraction.text import CountVectorizer

    # 创建CountVectorizer对象
    vectorizer = CountVectorizer(max_features=1024)
    words = vectorizer.fit_transform(raw_texts)
    bow = words.toarray()
    bow = torch.Tensor(bow)
    print('bow', bow.size())

    return bow

def get_tf_idf_by_texts(texts, known_mask, test_mask, max_features = 1433, use_tokenizer = False):
    if known_mask == None and test_mask == None:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = tf_idf_vec.fit_transform(texts)
        torch_feat = torch.FloatTensor(X.todense())
        norm_torch_feat = F.normalize(torch_feat, dim = -1)
        return torch_feat, norm_torch_feat
    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir = "/tmp")
        tf_idf_vec = TfidfVectorizer(analyzer="word", max_features=500, tokenizer=lambda x: tokenizer.tokenize(x, max_length=512, truncation=True))
        text_known = texts[known_mask]
        text_test = texts[test_mask]
    else:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        text_known = texts[known_mask]
        text_test = texts[test_mask]
    x_known = tf_idf_vec.fit_transform(text_known)
    x_test = tf_idf_vec.transform(text_test)
    x_known = torch.FloatTensor(x_known.todense())
    x_test = torch.FloatTensor(x_test.todense())
    dim = x_known.shape[1]
    torch_feat = torch.zeros(len(texts), dim)
    torch_feat[known_mask] = x_known
    torch_feat[test_mask] = x_test
    norm_torch_feat = F.normalize(torch_feat, dim = -1)
    return torch_feat, norm_torch_feat

def load_secret():
    with open('secret.yaml') as f:
        secret = yaml.safe_load(f)
    return secret
def get_word2vec(raw_texts):
    raw_text = [[ x for x in line.lower().split(' ') if x.isalpha()] for line in raw_texts]
    w2v_path = load_secret()['word2vec']['path']
    word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary = True)
    vecs = []
    for sentence in raw_text:
        tokens = [x for x in sentence if x.isalpha()]
        word_vectors = [word2vec[word] for word in tokens if word2vec.key_to_index.get(word, None)]
        if len(word_vectors) == 0:
            vecs.append(np.zeros(300))
        else:
            sentence_vectors = np.mean(word_vectors, axis = 0)
            vecs.append(sentence_vectors)
    vecs = np.vstack(vecs)
    vecs = torch.FloatTensor(vecs)
    return vecs

def _compute_popularity_property(graph_nx, ascending=True):
    direction = -1 if ascending else 1
    property_values = direction * np.array(list(A.pagerank(graph_nx).values()))
    return property_values


def _compute_locality_property(graph_nx, ascending=True):
    num_nodes = graph_nx.number_of_nodes()
    pagerank_values = np.array(list(A.pagerank(graph_nx).values()))

    personalization = dict(zip(range(num_nodes), [0.0] * num_nodes))
    personalization[np.argmax(pagerank_values)] = 1.0

    direction = -1 if ascending else 1
    property_values = direction * np.array(
        list(A.pagerank(graph_nx, personalization=personalization).values())
    )
    return property_values


def _compute_density_property(graph_nx, ascending=True):
    direction = -1 if ascending else 1
    property_values = direction * np.array(
        list(A.clustering(graph_nx).values())
    )
    return property_values