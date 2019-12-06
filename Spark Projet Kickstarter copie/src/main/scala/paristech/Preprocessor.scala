package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    import org.apache.spark.sql.{DataFrame, SparkSession, functions, Column}
    import org.apache.spark.sql.functions.lower
    import spark.implicits._
    import org.apache.spark.sql.functions.udf
    import org.apache.spark.sql.functions.{concat, lit}

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    val df2: DataFrame = dfCasted.drop("disable_communication")

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    def days_of_campaign(launched_at: Int, deadline: Int) : Int = {
      (deadline - launched_at) / 86400
    }

    val days_campaignUdf = udf(days_of_campaign _)

    val dfDaysCampaign: DataFrame = dfCountry
      .withColumn("days_campaign", days_campaignUdf($"launched_at", $"deadline"))

    def hours_of_prepa(created_at: Int, launched_at: Int) : Double = {
      (launched_at.toDouble - created_at.toDouble)/3600
    }

    val hours_prepUdf = udf(hours_of_prepa _)

    val dfHoursPrepa: DataFrame = dfDaysCampaign
      .withColumn("hours_prepa", hours_prepUdf($"created_at", $"launched_at"))

    val dfHoursPrepaClean : DataFrame = dfHoursPrepa.drop("created_at","launched_at","deadline")

    val dfHoursPrepaLower : DataFrame = dfHoursPrepaClean
      .withColumn("desc_lower",lower($"desc"))
      .withColumn("keywords_lower",lower($"keywords"))
      .withColumn("name_lower",lower($"name"))

    val dfConcat: DataFrame = dfHoursPrepaLower
      .withColumn("text",concat($"name_lower",lit(" "),$"desc_lower",lit(" "),$"keywords_lower"))

    val dfClean: DataFrame = dfConcat
      .na.fill(-1,Seq("days_campaign"))
      .na.fill(-1,Seq("hours_prepa"))
      .na.fill(-1,Seq("goal"))
      .na.fill("unknown",Seq("country2"))
      .na.fill("unknown",Seq("currency2"))

    dfClean.write.mode("overwrite").parquet("src/main/resources/preprocessed/output.parquet")
  }
}
