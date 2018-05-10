#-*- coding:utf-8 -*-
from __future__ import division
from tqdm import tqdm
import jieba.posseg as pseg
import cPickle
import json
import sys
import codecs
import chardet
import math
reload(sys)
sys.setdefaultencoding("utf-8")

positive = dict()
negative = dict()
positive["length_size"] = 0
negative["length_size"] = 0
stopwords = set()
punction = set()
def load_stopwords(file_name):
	with open(file_name) as readme:
		for l in readme.readlines():
			stopwords.add(l.strip())
			
load_stopwords("data\\stopwords.txt")

def predict(x,y,ngram = 2):
	global positive
	global negative
	
	res = dict()
	try:
		title = y.split("###")[0]
		post = y.split("###")[1]
	except:
		title = ""
		post = y
	title = pseg.cut(title)
	post = pseg.cut(post)
	title = [[o.word,o.flag] for o in title if o.flag.startswith("v") or o.flag == "n" ]
	post = [[o.word,o.flag] for o in post if o.flag.startswith("v") or o.flag == "n"]
	for i in range(len(post)- ngram + 1):
		continue_flag = True
		for wd in post[i:i+ngram]:
			if wd[1].startswith("v"):continue_flag = False
		if continue_flag:continue
		words = [w[0] for w in post[i:i+ngram] if not w[0] in stopwords]
		words = "".join(words)
		if words == "":continue
		if words not in res.keys():
			res[words] = 1
		else:
			res[words] += 1
			
	for i in range(len(title)- ngram + 1):
		continue_flag = True
		for wd in title[i:i+ngram]:
			if wd[1].startswith("v"):continue_flag = False
		if continue_flag:continue
		words = [w[0] for w in title[i:i+ngram] if not w[0] in stopwords]
		words = "".join(words)
		if words == "":continue
		if words not in res.keys():
			res[words] = 2
		else:
			res[words] += 2
	
	assert x in ('0','-1','1'),"x should in -1 0 1"
	if x == '0':
		positive["length_size"] += 1
		
		for word in res.keys():
			if res[word] < 0:continue #È¥³ýµÍÆµ´Ê
			if not word in positive.keys():
				positive[word] = 1
			else:
				positive[word] += 1
			assert positive[word] <= positive["length_size"] ,"should be no more than positive total"
			
	elif  x == '-1':
		negative["length_size"] += 1
				
		for word in res.keys():
			if res[word] < 0:continue #È¥³ýµÍÆµ´Ê
			if not word in negative.keys():
				negative[word] = 1
			else:
				negative[word] += 1
			assert negative[word] <= negative["length_size"],"should be no more than negative total"
	else:
		pass
		#raise BaseException("invalid mark: " + x)


with open(r"data\chat.txt") as readme:
	for l in tqdm(readme.readlines()):
		l = l.replace("\n","").replace("\xef\xbb\xbf","")
		x = l.split("\t")[0].strip()
		y = l.split("\t")[1].strip()
		predict(x,y)

fetures  = dict()
print "find in negative " + "*"*20
for word in tqdm(negative.keys()):
	A = negative[word]
	try:
		B = positive[word]
	except KeyError:
		B = 0
	C = negative["length_size"] - A
	D = positive["length_size"] - B
	if A == negative["length_size"] and B == positive["length_size"]:continue
	score = float(pow((A*D - B*C),2) / ((A+B) * (C+D)))
	fetures[word] = score
	
print "find in positive " + "*"*20
for word in tqdm(positive.keys()):
	if word in fetures.keys():continue
	A = positive[word]
	try:
		B = negative[word]
	except KeyError:
		B = 0
	C = positive["length_size"] - A
	D = negative["length_size"] - B
	if A == positive["length_size"] and B == negative["length_size"]:continue
	score = float(pow((A*D - B*C),2) / ((A+B) * (C+D)))
	fetures[word] = score
	
f = open(r"model\fetures_chat.txt","w")
cPickle.dump(fetures,f)
f.close()

		
