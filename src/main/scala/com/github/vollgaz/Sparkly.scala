package com.github.vollgaz

import org.apache.spark.sql.SparkSession

trait Sparkly {
    val spark: SparkSession = SparkSession.builder()
        .master("local[*]")
        .appName("test")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

}
