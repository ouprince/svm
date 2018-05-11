# -*- coding:utf-8 -*-
import os,sys
import json
reload(sys)
import requests
sys.setdefaultencoding("utf-8")
import cPickle
import jieba
import jieba.posseg as pseg
import numpy
from tqdm import tqdm
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report
curdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.join(curdir,os.path.pardir)
sys.path.append(rootdir)

from chinese_whispers.app.common.similarity import compare

#加载自定义词典
jieba.load_userdict(os.path.join(curdir,"data","word_zidingyi.txt"))

fetures_word = []
with open("data/fetures_chat.txt") as readme:
    for w in readme.readlines():
        fetures_word.append(w.replace("\n",""))
		
print "len(fetures_word) = %d" %len(fetures_word)

y_mark = []
y_vector = []

				
def load_vector(file_name,ngram = 1,mark_one = ['0'],mark_two = ['-1']):
    global y_mark
    global y_vector
    def similarity(s1,s2):
        try: return max(compare(s1,s2, seg=True , version = '2.0'),compare(s2,s1,seg = True,version = '2.0'))
        except:return 0.0

    print "loading vector ..."
    with open(file_name) as readme:
        for l in tqdm(readme.readlines()):
            l = l.replace("\n","").replace("\xef\xbb\xbf","")
            x = l.split("\t")[0]
            y = l.split("\t")[1]
            try:
                title = y.split("###")[0]
                post = y.split("###")[1]
            except:
                title = ""
                post = y
            assert x in ('0','1','-1'),"x should be -1 0 1"
            if x in mark_one:
                y_mark.append(0)
            elif x in mark_two:
                y_mark.append(1)
            else:
                continue
                #raise BaseException("invalide mark")
	
            y = title + post
            y = pseg.cut(y)
            y = [o.word for o in y if o.flag.startswith("v") or o.flag == "n"]
            vector = []
            for w in fetures_word:
                max_score = 0.0
                for i in range(len(y)- ngram + 1):
                    word = "".join(y[i:i + ngram])
                    max_score = max(similarity(w,word),max_score)
                if max_score < 0.2:max_score = 0.0
                elif max_score > 0.8:max_score = 1.0
                vector.append(max_score)
            y_vector.append(vector)
	
if __name__ == "__main__":
    ceshi = pseg.cut("你好为什么")
    for w in ceshi:
        print w.word,w.flag

    canshu = {"mark_one":["0"],"mark_two":["-1","1"],"ngram":1}
    load_vector("data/chat.txt",**canshu)
    assert len(y_vector) == len(y_mark),"vector should be same as long as mark"
    while True:
        y_train, y_test, y_train_mark, y_test_mark = train_test_split(y_vector,y_mark,test_size = 0.1)
        clf = svm.SVC(C = 0.7, kernel = 'rbf', degree = 3, coef0 = 0.0,
                            probability=False, tol=0.001, cache_size=1000, class_weight='balanced', verbose=False,
                                                 max_iter=-1, random_state=None)
        clf.fit(y_train,y_train_mark) # 训练模型
        joblib.dump(clf,"svm/train.raw.data.pkl") # 保存模型 
        result = clf.predict(y_test)
        labels = [0,1]
        target_names = ["non_negative","negative"]
        positive_rate = classification_report(y_test_mark,result,labels = labels,target_names = target_names).split("\n")[2].split()[1]
        negative_rate = classification_report(y_test_mark,result,labels = labels,target_names = target_names).split("\n")[3].split()[1]

        print classification_report(y_test_mark,result,labels = labels,target_names = target_names) # 评估分类效果

        if float(negative_rate) >= 0.8 and float(positive_rate) >= 0.8:
            break

