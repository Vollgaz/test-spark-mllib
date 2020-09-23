package com.github.vollgaz

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.RFormula

object ExampleLogisticRegression extends App with Sparkly {
    val df = spark.read.json("data/simple-ml.json")
    df.orderBy("value2").show()


    val supervised = new RFormula()
        .setFormula("lab ~ . + color:value1 + color:value2")

    val fittedRF = supervised.fit(df)
    val preparedDF = fittedRF.transform(df)
    preparedDF.show()

    val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))

    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    println(lr.explainParams())

}
