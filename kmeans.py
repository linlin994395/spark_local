# -*- coding: utf-8 -*-
"""
    function: kmeans
    author: xuel
    email: xuel@
    create date: 2019/3/18
    Software: PyCharm 
    copyright: @2019 Startimes.Co.Ltd. All rights reserved.
"""

from pyspark.sql import Row
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


def get_local_spark_context():
    sc = SparkContext(appName="local_context")
    # 必须加，否则rdd到dataframe的转换会报错
    sqlContext = SQLContext(sc)
    return sc


def f(x):
    rel = dict()
    rel['features'] = Vectors.dense(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
    return rel


if __name__ == '__main__':
    sc = get_local_spark_context()
    # df = sc.textFile("iris.csv").map(lambda line: line.split(','))\
    #     .map(lambda p: Row(duran=float(p[0]), rju=float(p[1]))).toDF()
    df = sc.textFile("data/iris.csv").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
    df.show()
    df.printSchema()
    kmeans_model = KMeans().setK(3).setFeaturesCol('features').setPredictionCol('prediction').fit(df)

    results = kmeans_model.transform(df).collect()
    for item in results:
        print(str(item[0]) + ' is predcted as cluster' + str(item[1]))

    # 打印出聚类中心的坐标
    results2 = kmeans_model.clusterCenters()
    for item in results2:
        print(item)

    print(kmeans_model.computeCost(df))
