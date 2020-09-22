
ThisBuild / scalaVersion     := "2.12.12"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.example"
ThisBuild / organizationName := "example"

lazy val root = (project in file("."))
  .settings(
    name := "test-spark-mllib",
    libraryDependencies := Seq(
        "org.apache.spark" %% "spark-sql" % "3.0.1",
        "org.apache.spark" %% "spark-mllib" % "3.0.1",
        "org.scalatest" %% "scalatest" % "3.2.0"

    )
  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
