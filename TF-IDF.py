# -*- coding: utf-8 -*-
"""
    function: LogisticRegression
    author: xuel
    email: xuel@
    create date: 2019/3/18
    Software: PyCharm 
    copyright: @2019 Startimes.Co.Ltd. All rights reserved.
"""

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark import SparkContext
from pyspark.sql import SQLContext


if __name__ == '__main__':
    sc = SparkContext(appName="tf_idf")
    sc._jsc.hadoopConfiguration().set('my.mapreduce.setting', 'someVal')
    sqlContext = SQLContext(sc)
    # sc = get_local_spark_context()
    sentenceData = sqlContext.createDataFrame(
        [(0, "I heard about Spark and I love Spark"), (0, "I wish Java could use case classes"),
         (1, "Logistic regression models are neat")]).toDF("label", "sentence")

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)
    wordsData.show(truncate=300)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=30)
    featurizedData = hashingTF.transform(wordsData)
    featurizedData.show(truncate=500)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)

    rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show(truncate=500)
