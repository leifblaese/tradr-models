
lazy val libdeps = Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
  "org.nd4j" % "nd4j-native-platform" % "0.9.1",
  "tradr" %% "tradr-common" % "0.0.5",
  "com.datastax.oss" % "java-driver-core" % "4.0.0-alpha1",
  "com.datastax.oss" % "java-driver-parent" % "4.0.0-alpha1",
  "tradr" %% "tradr-cassandra-connector" % "0.0.2"
)


lazy val root = (project in file("."))
  .configs(IntegrationTest)
  .settings(Seq(
    name := "tradr-models",
    organization := "tradr",
    scalaVersion := "2.12.3",
    crossScalaVersions := Seq("2.11.11", "2.12.3"),
    version := "0.0.2",
    libraryDependencies ++= libdeps,
    assemblyJarName in assembly :=  s"${name.value}_${scalaVersion.value}-${version.value}.jar"
  ))

