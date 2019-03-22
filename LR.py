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
    rel['features'] = Vectors.dense(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
    rel['label'] = str(x[4])
    return rel


if __name__ == '__main__':
    sc = get_local_spark_context()
    data = sc.textFile("data/iris.csv").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
    # data.printSchema()

    trainingData, testData = data.randomSplit([0.7, 0.3])
    labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(data)

    lr = LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(50)

    labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    # pipeline
    lrPipeline = Pipeline().setStages([labelIndexer, featureIndexer, lr, labelConverter])

    paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.2, 0.8]).addGrid(lr.regParam, [0.01, 0.1, 0.5]).build()

    # 构建针对整个机器学习工作流的交叉验证类，定义验证模型、参数网格，以及数据集的折叠数，并调用fit方法进行模型训练
    cv = CrossValidator().setEstimator(lrPipeline)\
        .setEvaluator(MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction"))\
        .setEstimatorParamMaps(paramGrid).setNumFolds(3)
    cvModel = cv.fit(trainingData)

    lrPredictions = cvModel.transform(testData)
    lrPreRel = lrPredictions.select("predictedLabel", "label", "features", "probability").collect()
    for item in lrPreRel:
        print(str(item['label']) + ',' + str(item['features']) + '-->prob=' + str(item['probability'])
              + ',predictedLabel' + str(item['predictedLabel']))

    # evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
    # lrAccuracy = evaluator.evaluate(lrPredictions)
    # print(lrAccuracy)
    #
    # bestModel = cvModel.bestModel
    # lrModel = bestModel.stages[2]
    # print("Coefficients: " + str(lrModel.coefficientMatrix) + "Intercept: " + str(lrModel.interceptVector)
    #       + "numClasses: " + str(lrModel.numClasses) + "numFeatures: " + str(lrModel.numFeatures))
    #
    # lr.explainParam(lr.regParam)
    # lr.explainParam(lr.elasticNetParam)