package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{IDF}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.tuning.CrossValidator



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    val dfClean_1: DataFrame= spark.read.parquet("src/main/resources/preprocessed/output.parquet")

    val dfClean = dfClean_1.filter($"text".isNotNull)

    //création du tokenizer pour chaque mot
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //création du remover de stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_filtered")

    //partie TF
    val vectorizer = new CountVectorizer()
      .setInputCol("tokens_filtered")
      .setOutputCol("tokens_vectorized")

    //partie IDF
    val idf = new IDF()
      .setInputCol("tokens_vectorized")
      .setOutputCol("tfidf")

    // convertir les variables textuelles country et currency en numérique

    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // One hot encoding

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed","currency_indexed"))
      .setOutputCols(Array("country_onehot","currency_onehot"))

    // Mettre les données sous une forme utilisable par SparkML

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa","goal","country_onehot","currency_onehot"))
      .setOutputCol("features")

    //Créer le modèle de Régression Logistique
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    //Création du pipeline

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,remover,vectorizer,idf,country_indexer,currency_indexer,encoder,vectorAssembler,lr))

    //Découpage du Dataframe en train et test

    val Array(training, test) = dfClean.randomSplit(Array(0.7, 0.3))

    // Entraînement et sauvegarde du modèle

    val model = pipeline.fit(training)

    // Test du modèle
    val dfWithSimplePredictions = model.transform(test)
    dfWithSimplePredictions.groupBy("final_status","predictions").count.show()

    // Evaluation des prédictions
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")
    val f1_score = evaluator.evaluate(dfWithSimplePredictions)
    println("Le score f1 est donc " + f1_score)

    //Réglage des hyperparamètres
    val paramGrid = new ParamGridBuilder()
      .addGrid(vectorizer.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    //Validation croisée
    val cross_val = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.70)

    //entraînement du nouveau modèle optimisé
    val model_opt = cross_val.fit(training)

    // prédictions du modèle optimisé et nouveau score f1
    val dfWithPredictions = model_opt.transform(test)
    val f1_opt = evaluator.evaluate(dfWithPredictions)
    println("Le Score f1 après la validation croisée est donc " + f1_opt)

    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    println("hello world ! from Trainer")


  }
}
