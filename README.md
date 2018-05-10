## 链接
	https://github.com/ouprince/svm.git
	
## 说明
	svm_article 是关于金融咨询类的利好/利空分析
	svm_comment 是关于评论的非吐槽/吐槽分析
	
	svm_article 和 svm_comment 文件夹里面的代码说明:
	(1)fetures.py  利用卡方差原理提取出特征词并保存
	(2)vector.py   利用卡方提取的特征词生成向量并生成 svm 模型并保存
	(3)predict.py  加载训练好的 svm 模型实现预测
	
	svm_article 和 svm_comment 使用的算法一样，只是训练的语料不同
	
## 备注
	svm_article 数据说明
	训练语料是基于 7万 多条金融咨询的 利好/利空 文章（由于github 文件限制，可在我的百度网盘找到- 链接：https://pan.baidu.com/s/1YvecNYmSTKeW3CpCJbKBZQ 密码：maad）
	这 7万 多的语料提取的卡方特征词: data/fetures_oyp.txt 训练好的模型保存在: svm/train.raw.data.pkl
	而在里面的 svm/opti_negative.pkl 则是基于 data/tr.txt 里面的 30 篇正负性文章所训练的 svm 模型
	
	svm_comment 数据说明
	基于某个 app 的评论分为 非负向情感/负向情感 两类进行的训练，自己标注了其中 600 条评论分为 0 中性，-1 负面。
	由于缺少正面评论语料，因此模型主要是分类 中性/负性 两类评论。
	语料: data\chat.txt 卡方提取的特征词: fetures_chat.txt 训练模型保存在: svm/train.raw.data.pkl
	

