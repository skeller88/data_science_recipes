# def generate_ngrams(text, stopwords, n_gram=1):
#     token = [token for token in text.lower().split(' ') if token != '' if token not in stopwords]
#     tozip = []
#     for i in range(n_gram):
#         print(i)
#         t = token[i:]
#         print(t)
#         tozip.append(t)
#     ngrams = zip(*[token[i:] for i in range(n_gram)])
#     print(list(ngrams))
#     return [' '.join(ngram) for ngram in ngrams]
#
#

def generate_ngrams(text, stopwords, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    print(list(ngrams))
    return [' '.join(ngram) for ngram in ngrams]

print('8', generate_ngrams("the cow is", stopwords=[], n_gram=2))
#
