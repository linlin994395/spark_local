# -*- coding: utf-8 -*-
"""
    function: als_cf
    author: xuel
    email: xuel@
    create date: 2019/3/18
    Software: PyCharm 
    copyright: @2019 Startimes.Co.Ltd. All rights reserved.
"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


def get_local_spark_context():
    sc = SparkContext(appName="local_context")
    # 必须加，否则rdd到dataframe的转换会报错
    sqlContext = SQLContext(sc)
    return sc


def f(x):
    rel = dict()
    rel['userId'] = int(x[0])
    rel['movieId'] = int(x[1])
    rel['rating'] = float(x[2])
    rel['timestamp'] = float(x[3])
    return rel


if __name__ == '__main__':
    sc = get_local_spark_context()
    ratings_df = sc.textFile("data/sample_movielens_ratings.txt").map(lambda line: line.split('\t'))\
        .map(lambda p: Row(**f(p))).toDF()
    # ratings_df.show()
    # ratings_df.printSchema()
    training, test = ratings_df.randomSplit([0.8, 0.2])

    alsExplicit = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
    alsImplicit = ALS(maxIter=5, regParam=0.01, implicitPrefs=True, userCol="userId", itemCol="movieId",
                      ratingCol="rating")

    # 训练
    modelExplicit = alsExplicit.fit(training)
    modelImplicit = alsImplicit.fit(training)

    # modelExplicit.setColdStartStrategy("drop")
    # modelImplicit.setColdStartStrategy("drop")

    # 预测
    predictionsExplicit = modelExplicit.transform(test)
    predictionsImplicit = modelImplicit.transform(test)

    predictionsExplicit = predictionsExplicit.na.drop()
    predictionsImplicit = predictionsImplicit.na.drop()
    # print(predictionsExplicit.printSchema())
    # print(predictionsImplicit.printSchema())

    predictionsExplicit.show()
    predictionsImplicit.show()

    evaluator = RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")

    rmseExplicit = evaluator.evaluate(predictionsExplicit)
    rmseImplicit = evaluator.evaluate(predictionsImplicit)
    print("Explicit:Root-mean-square error = " + str(rmseExplicit))
    print("Explicit:Root-mean-square error = " + str(rmseImplicit))

    # Generate top 10 movie recommendations for each user
    userRecs = modelExplicit.recommendForAllUsers(10)
    print(userRecs)
    # Generate top 10 user recommendations for each movie
    movieRecs = modelExplicit.recommendForAllItems(10)
    print(movieRecs)

    # Generate top 10 movie recommendations for a specified set of users
    users = ratings_df.select(alsExplicit.getUserCol).distinct().limit(3)
    userSubsetRecs = modelExplicit.recommendForUserSubset(users, 10)
    print(userSubsetRecs)
    # Generate top 10 user recommendations for a specified set of movies
    movies = ratings_df.select(alsExplicit.getItemCol).distinct().limit(3)
    movieSubSetRecs = modelExplicit.recommendForItemSubset(movies, 10)
    print(movieSubSetRecs)
