# -*- coding: utf-8 -*-
"""
    function: LogisticRegression
    author: xuel
    email: xuel@
    create date: 2019/3/18
    Software: PyCharm 
    copyright: @2019 Startimes.Co.Ltd. All rights reserved.
"""

from pyspark.ml.feature import CountVectorizer
from pyspark import SparkContext
from pyspark.sql import SQLContext


if __name__ == '__main__':
    sc = SparkContext(appName="CountVectorizer")
    sc._jsc.hadoopConfiguration().set('my.mapreduce.setting', 'someVal')
    sqlContext = SQLContext(sc)
    # Input data: Each row is a bag of words with a ID.
    df = sqlContext.createDataFrame([
        (0, "a b c".split(" ")),
        (1, "a b b c a".split(" "))
    ], ["id", "words"])

    # fit a CountVectorizerModel from the corpus.
    cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=2, minDF=2.0)
    model = cv.fit(df)
    result = model.transform(df)
    result.show(truncate=300)
