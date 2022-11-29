"""
train word2vec on hebrew wikipedia.
note that the hebrew wikipedia is a bias data set...

this script contains 2 parts -
get text:
    hebrew wikipedia is about 800MB at zip condition.
    download the wikipedia content as xml.zip and parse it with gensim.
train word2vec:
    give the text to the word2vec and train it.

downloading the xml of the hebrew wikipedia:
        https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2
    as alternative, you can download only 10% of it from
        https://dumps.wikimedia.org/hewiki/latest/hewiki-20211001-pages-articles-multistream1.xml-p1p68044.bz2
    put the downloaded xml at ~/Downloads/gensim_heb_wiki
    no need to unzip it...

run time:
    about 1 hour.
    the sampled xml will take about 10 minutes

you can also find the wiki.word2vec.model file here. can use it by:
    from gensim.models.word2vec import Word2Vec
    word2vec = Word2Vec.load("wiki.word2vec.model")
    word2vec.wv.most_similar...
"""
import os
import logging

import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
import pandas as pd

# tell word2vec to update on the progress
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


model_dump_folder = fr'{os.path.expanduser("~")}\Downloads\gensim_heb_wiki'
model_dump_wiki_corpus = f'{model_dump_folder}/wiki.corpus.model'
model_dump_word2vec = f'{model_dump_folder}/wiki.word2vec.model'
# download this bz2 file from https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2
wiki_xml = fr'{model_dump_folder}\hewiki-20211001-pages-articles-multistream1.xml-p1p68044.bz2'
wiki_xml = fr'{model_dump_folder}\hewiki-latest-pages-articles.xml.bz2'


print('parse wiki xml')
if os.path.isfile(model_dump_wiki_corpus):
    print('reading wiki corpus model')
    wiki = WikiCorpus.load(model_dump_wiki_corpus)
else:
    wiki = WikiCorpus(wiki_xml, lemmatize=False)  # 10 minutes for whole hebrew wiki (800MB)
    wiki.save(model_dump_wiki_corpus)


print('building word2vec')
if os.path.isfile(model_dump_word2vec):
    print('reading word2vec model')
    word2vec = Word2Vec.load(model_dump_word2vec)
else:
    class MySentences(object):
        def __iter__(self):
            for text in wiki.get_texts():
                yield text


    sentences = MySentences()
    params = dict(size=300,
                  window=10,
                  min_count=40,
                  workers=multiprocessing.cpu_count() + 2,
                  sample=1e-3)
    word2vec = Word2Vec(sentences, **params)
    word2vec.save(model_dump_word2vec)


print('playground')
# now some playing around
pd.DataFrame(word2vec.wv.most_similar('משחקים', topn=5), columns=['word', 'score'])
# returning למשחקים, שחקנים, משחרי, טורנירים, משחק

pd.DataFrame(word2vec.wv.most_similar('במכונית', topn=5), columns=['word', 'score'])
# returning ברכב, הנהג, בנסיעה, האוטובוס, באקדח

word2vec.wv.most_similar_to_given('במכוניתו', ['רכב', 'מכונית', 'שולחן', 'הלך', 'שלום'])
# returning מכונית

word = 'במכוניתו'
df = pd.DataFrame()
df['other_word'] = ['במכוניתו', 'רכב', 'מכונית', 'שולחן', 'הלך', 'שלום']
df['distance'] = word2vec.wv.distances(word, df.other_word.tolist())
# returning score for each specific given option
