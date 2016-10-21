# -*- encoding:utf-8 -*-
import lda
import numpy as np
import lda.datasets

# 载入文档-词矩阵
# reuters为路透社新闻
X = lda.datasets.load_reuters()
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))
print(X[:5, :5])


# 词汇
vocab = lda.datasets.load_reuters_vocab()
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))
print(vocab[:6])

# 标题/文档
titles = lda.datasets.load_reuters_titles()
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))
print(titles[:2])  # 前两篇文章的标题

# 训练模型
model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)
model.fit(X)

# 每个topic的词矩阵
topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

# 显示每个话题的topn的词
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:n]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

# 每个文档的topic矩阵
doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))

# 显示前10个文档的top1的话题
for n in range(10):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}".format(n, topic_most_pr))