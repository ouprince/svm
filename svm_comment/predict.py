# -*- coding:utf-8 -*-
import os,sys
import json
import numpy
reload(sys)
import jieba
import jieba.posseg as pseg
from sklearn.externals import joblib
curdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.join(curdir,os.path.pardir)

sys.path.append(rootdir)
from chinese_whispers.app.common.similarity import compare
#from gensim.models import KeyedVectors

#model = KeyedVectors.load_word2vec_format(os.path.join(curdir,"model","20171129.bin"), \
                                            #binary = True, encoding = "utf-8", unicode_errors = "ignore")

# 加载自定义词典
jieba.load_userdict(os.path.join(curdir,"data","word_zidingyi.txt"))

fetures_word = []
with open(os.path.join(curdir,"data","fetures_chat.txt")) as readme:
    for w in readme.readlines():
        fetures_word.append(w.replace("\n",""))


def vectored(**kwargs):
    comment = kwargs["comment"]
    try:
        ngram = kwargs["ngram"]
    except:
        ngram = 1

    def similarity(s1,s2):
        try: return max(compare(s1,s2, seg=True , version = '2.0'),compare(s2,s1,seg = True,version = '2.0'))
        except:return 0.0

    print "loading vector ..."
    comment = pseg.cut(comment)
    comment = [o.word for o in comment if o.flag.startswith('v') or o.flag in ('a','n')]

    vector = []
    for w in fetures_word:
        max_score = 0.0
        for i in range(len(comment)- ngram + 1):
            word = "".join(comment[i:i + ngram])
            max_score = max(similarity(w,word),max_score)
        if max_score < 0.2:max_score = 0.0
        elif max_score > 0.8:max_score = 1.0
        vector.append(max_score)

    return vector

def predict(comment,ngram = 1):
    vector_comment = vectored(comment = comment,ngram = ngram)
    clf = joblib.load(os.path.join(curdir,"svm","train.raw.data.pkl"))
    result = clf.predict([vector_comment])[0]
    score = abs(clf.decision_function([vector_comment])[0])
    if result == 0:
        return json.dumps({"result":"non_negative","score":float(score)})
    else:
        return json.dumps({"result":"negative","score":float(score)})

if __name__ == "__main__":
    print predict(comment = "垃圾玩意")
