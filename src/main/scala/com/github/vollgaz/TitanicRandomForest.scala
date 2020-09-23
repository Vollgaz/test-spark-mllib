package com.github.vollgaz

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{FeatureHasher, IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

object TitanicRandomForest extends App with Sparkly {

    val trainDS = spark.read
        .option("header", value = true)
        .option("sep", ",")
        .option("inferSchema", true)
        .csv("data/titanic/train.csv")


    val testDS = spark.read
        .option("header", value = true)
        .option("sep", ",")
        .option("inferSchema", true)
        .csv("data/titanic/test.csv")

    println("========== raw train data")
    //trainDS.ls



    val ftHasher = new VectorAssembler()
        .setInputCols(Array("Age", "SibSp", "Parch", "Fare"))
        .setOutputCol("features")


    val ftTrainDS = ftHasher.transform(trainDS)

    println("========== featurised train data")
    //ftTrainDS.ls


    val ftTestDS = ftHasher.transform(testDS)
    println("========== featurised test data")
   // ftTestDS.ls



    val rf: RandomForestClassifier = new RandomForestClassifier()
        .setLabelCol("Survived")
        .setFeaturesCol("features")
        .setNumTrees(10)


    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(rf))

    val model: PipelineModel = pipeline.fit(ftTrainDS)

    val predictions = model.transform(ftTestDS)

    predictions.show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")


    scala.io.StdIn.readChar()
    spark.stop()


    implicit class DataFrameExtended(e: DataFrame) {

        def ls: Unit = {
            e.printSchema()
            e.show(5, false)
        }
    }

}