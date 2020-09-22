package com.github.vollgaz

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.RandomForest

object TitanicRandomForest extends App with Sparkly {

    val trainDS = spark.read
        .option("header", value = true)
        .option("sep", ",")
        .option("inferSchema", true)
        .csv("data/titanic/train.csv")

    val indexer =  new StringIndexer()
    indexer.setInputCol()
    val testDS = spark.read
        .option("header", value = true)
        .option("sep", ",")
        .option("inferSchema", true)
        .csv("data/titanic/test.csv")

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainDS,
        numClasses,
        categoricalFeaturesInfo,
        numTrees,
        featureSubsetStrategy,
        impurity,
        maxDepth,
        maxBins)

    scala.io.StdIn.readChar()
    spark.stop()
}
