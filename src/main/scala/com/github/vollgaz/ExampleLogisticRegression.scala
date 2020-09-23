package com.github.vollgaz

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object ExampleLogisticRegression extends App with Sparkly {
    val df = spark.read.json("data/simple-ml.json")
    df.orderBy("value2").show()

    val Array(train, test) = df.randomSplit(Array(0.7, 0.3))

    val rForm = new RFormula()

    val lr = new LogisticRegression()
        .setLabelCol("label")
        .setFeaturesCol("features")


    val stages = Array(rForm, lr)

    val pipeline = new Pipeline().setStages(stages)

    val params = new ParamGridBuilder()
        .addGrid(rForm.formula, Array(
            "lab ~ . + color:value1",
            "lab ~ . + color:value1 + color:value2")
        )
        .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
        .addGrid(lr.regParam, Array(0.1, 2.0))
        .build()

    val evaluator = new BinaryClassificationEvaluator()
        .setMetricName("areaUnderROC")
        .setRawPredictionCol("prediction")
        .setLabelCol("label")

    val tvs = new TrainValidationSplit()
        .setTrainRatio(0.75) // also the default.
        .setEstimatorParamMaps(params)
        .setEstimator(pipeline)
        .setEvaluator(evaluator)

    val tvsFitted = tvs.fit(train)
    val evaluation = evaluator.evaluate(tvsFitted.transform(test))
    println(s" evaluator = $evaluation")

    val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
    val TrainedLR = trainedPipeline.stages(1).asInstanceOf[LogisticRegressionModel]
    val summaryLR = TrainedLR.summary
    println(s" history = ${summaryLR.objectiveHistory.mkString(", ")}")

    tvsFitted.write.overwrite().save("model/exampleLogisticRegression")
    scala.io.StdIn.readLine()
}
