# -*- coding: utf-8 -*-
"""
    function: word2vec
    author: xuel
    email: xuel@
    create date: 2019/3/18
    Software: PyCharm 
    copyright: @2019 Startimes.Co.Ltd. All rights reserved.
    interface refer to: http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.feature.Word2VecModel
"""

from pyspark.ml.feature import Word2Vec
from pyspark import SparkContext
from pyspark.sql import SQLContext


if __name__ == '__main__':
    sc = SparkContext(appName="word2vec")
    sc._jsc.hadoopConfiguration().set('my.mapreduce.setting', 'someVal')
    sqlContext = SQLContext(sc)

    documentDF = sqlContext.createDataFrame([
        ("Hi I heard about Spark".split(" "),),
        ("I wish Java could use case classes".split(" "),),
        ("Logistic regression models are neat".split(" "),)
    ], ["text"])
    documentDF.show(truncate=300)

    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)

    # 得到每个词的及其训练得出的词向量
    word_vectors = model.getVectors()
    word_vectors.show(truncate=300)

    # 得到与某个词最近的几个词及相应的相似度
    # synonyms = model.findSynonyms("heard", 2)
    # synonyms.show(truncate=300)
    # synonyms.printSchema()

    # 得到整个文档的向量化表示
    # result = model.transform(documentDF)
    # result.show(truncate=300)
    # for row in result.collect():
    #     text, vector = row
    #     print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
