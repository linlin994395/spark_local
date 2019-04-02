# -*- coding: utf-8 -*-
"""
    function: LogisticRegression
    author: xuel
    email: xuel@
    create date: 2019/3/18
    Software: PyCharm 
    copyright: @2019 Startimes.Co.Ltd. All rights reserved.
"""

from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.feature import HashingTF,Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


def get_local_spark_context():
    sc = SparkContext(appName="local_context")
    # 必须加，否则rdd到dataframe的转换会报错
    sqlContext = SQLContext(sc)
    return sc


def f(x):
    rel = {}
    rel['label'] = str(x[4])
    return rel

def ha(x,tags_set):
    if x in tags_set:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sc = get_local_spark_context()
    tags_set = ['Iris-virginica']
    data = sc.textFile("data/iris.csv").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
    data.show()
    list = data.select("label").distinct().collect()
    result = list.rdd.map(lambda row: [ha(row[0],tags_set)])
    print(list)
