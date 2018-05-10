# -*- coding:utf-8 -*-
# 加载训练的模型并使用
import os,sys
import json
reload(sys)
sys.setdefaultencoding("utf-8")
import jieba.posseg as pseg
from sklearn.externals import joblib
curdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.join(curdir,os.path.pardir)
sys.path.append(rootdir)
from chinese_whispers.app.common.similarity import compare

fetures_word = []
with open(os.path.join(curdir,"data","fetures_oyp.txt")) as readme:
    for w in readme.readlines():
        fetures_word.append(w.replace("\n",""))


def vectored(**kwargs):
    title= kwargs["title"]
    post = kwargs["post"]      
    try:
        ngram = kwargs["ngram"]
    except:ngram = 2

    def similarity(s1,s2):
        try: return max(compare(s1,s2, seg=True , version = '2.0'),compare(s2,s1,seg = True,version = '2.0'))
        except:return 0.0

    print "loading vector ..."
    y = title + post
    y = pseg.cut(y)
    y = [o.word for o in y if o.flag.startswith("v") or o.flag == "n" ]
    vector = []
    for w in fetures_word:
        max_score = 0.0
        for i in range(len(y)- ngram + 1):
            word = "".join(y[i:i + ngram])
            max_score = max(similarity(w,word),max_score)
        if max_score < 0.2:max_score = 0.0
        elif max_score > 0.8:max_score = 1.0
        vector.append(max_score)

    return vector

def predict(title,post,ngram = 2):
    vecter_article = vectored(title = title,post = post,ngram = ngram)
    clf = joblib.load(os.path.join(curdir,"svm","train.raw.data.pkl"))
    result = clf.predict([vecter_article])[0]
    score = abs(clf.decision_function([vecter_article])[0])
    if result == 0:
        return json.dumps({"result":"positive","score":float(score)})
    elif result == 1:
        return json.dumps({"result":"negative","score":float(score)})

if __name__ == "__main__":
    print predict(title = "",post = "股市大涨，收益很显著")    

