package tradr.model.a3c

import com.typesafe.config.Config
import org.deeplearning4j.nn.gradient.Gradient
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.factory.Nd4j
import tradr.cassandraconnector.CassandraConnector
import tradr.common.PricingPoint
import tradr.common.predictor.PredictionResult
import tradr.common.trading._

import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}
import scala.util.Random




object A3CEstimator {




  /**
    * Compute the errors for a prediction and the resulting portfolio
    * @param predictionResult
    * @param portfolio
    * @param initialR
    * @param gamma
    * @param profit
    * @return
    */
  private def computeError(predictionResult: PredictionResult,
                   portfolio: Portfolio,
                   initialR: Double,
                   gamma: Double,
                   profit: Double) = {

    val probabilities = predictionResult
      .results("probabilities")
    val valuePrediction = predictionResult
      .results("valuePrediction")
      .head
    val R = gamma * initialR + profit

    val actionProbError = probabilities
      .map(Math.log)
      .map(_ * (R - valuePrediction))

    val valueFunError = Math.pow(R - valuePrediction, 2.0)

    (actionProbError, valueFunError, R)
  }


  /**
    * Compute errors for a certain time frame
    * @param gamma decay factor for expected returns over time
    * @param predictionPortfolioPairs Pairs of predicitons and the resulting portfolio changes
    * @return
    */
  private def computeAllErrors(gamma: Double,
                       predictionPortfolioPairs: Seq[(PredictionResult, Portfolio)]) = {

    val (predictionResult, portfolio) = predictionPortfolioPairs.last

    val lastCurrencyValue = portfolio.currencies(Currencies.EUR)
    val firstR = predictionResult.results("valuePrediction").head


    // For all portfolioPairs: Create the error
    val errors = predictionPortfolioPairs
      .indices
      .reverse
      .drop(1)
      .foldLeft((Array[Array[Double]](), Array[Double](), firstR)){
        case ((actionProbErrs, valueFunErrs, prevR), i) =>

          val (prediction, portfolio) = predictionPortfolioPairs(i)
          val profit = lastCurrencyValue - portfolio.currencies(Currencies.EUR)
          val (actionProbErr, valueFunErr, newR) = computeError(prediction, portfolio, prevR, gamma, profit)

          (actionProbErrs :+ actionProbErr, valueFunErrs :+ valueFunErr, newR)
      }
    (errors._1, errors._2)
  }


  /**
    *  Manually fit the network by computing the error, backprop'ing it and applying
    *  the updates
    *  @param network: A3CModel, the network end fit
    *  @param gamma: Decay factor for the reward
    *  @param predictionPortfolioPairs A sequence of predictions and the resulting portfolios
    *                                  with which the network should be trained
    */
  private def update(
              network: ComputationGraph,
              gamma: Double,
              predictionPortfolioPairs: Future[Seq[(PredictionResult, Portfolio)]],
              ): Future[ComputationGraph] = {

    predictionPortfolioPairs.map{
      pairs =>

        // Compute the errors for the whole sequence of predictions
        val (actionProbErrs, valueFunErrs) =
          computeAllErrors(gamma, pairs)

        val currentGradient: Gradient = network
          .backpropGradient(
            Nd4j.create(actionProbErrs),
            Nd4j.create(valueFunErrs)
          )

        val batchSize = network.batchSize()
        val updater = Option(network.getUpdater) // Is the updater set? NullpointerException posible?
        updater
          .map{
            upd =>
              upd.update(currentGradient, 0, batchSize)
              network
          }
          .getOrElse(throw new Exception("Updater of the computation graph not set"))

    }

  }



  /**
    * Find the portfolios and corresponding predictions.
    * We predict something and perform an action for which the portfolio changes.
    * That means for each portfolios and predictions have interleaving timestamps.
    * For each prediction we want end find the portfolio with the nearest time stamp and
    * vice versa. If there are no portfolios (because we did not perform a trade) we will
    * not consider this point in the input data. If there are more than one portfolios
    * we take the first one (this rule might change later on)
    * @param predictionsFuture Future of the predictions from cassandra
    * @param portfoliosFuture Future of the portfolios from cassandra
    * @param tradingFrequency Frequency of trading (e.g 1/minute)
    * @return
    */
  def mapPredictionsAndPortfolios(
                                  predictionsFuture: Future[Seq[PredictionResult]],
                                  portfoliosFuture: Future[Seq[(Long, Portfolio)]],
                                  tradingFrequency: Int): Future[Seq[(PredictionResult, Portfolio)]] = {
    implicit val ec: ExecutionContextExecutor = ExecutionContext.global
    predictionsFuture.flatMap {
      predictions =>

        val a: Seq[Future[(PredictionResult, Portfolio)]] = predictions
          .map { prediction =>
            val t = prediction.timestamp
            val portfolioOpt = portfoliosFuture
              .map { seq =>
                val filteredSeq =
                  seq.filter(s => s._1 >= t && s._1 < (t + tradingFrequency))

                filteredSeq.size match {
                  case 0 => None
                  case 1 => Some((prediction, filteredSeq.head._2))
                  case _ => Some((prediction, filteredSeq.head._2)) // For now, lets just take the first one. Usually we would want end have the mean or something
                }
              }
            portfolioOpt
          }
          .map(s => s.withFilter(o => o.isDefined).map(_.get))

        Future.sequence(a)

    }
  }




  /**
    *
    * Train the model start the predictions that occurred on the specific timeframe
    * @param start start of the timeframe used to update the network
    * @param end end of the time frame used to update the network
    * @param model Model to train
    * @param portfolioId Which portfolio to use
    * @param conf Typesafe config to use
    */
  def train(start: Long,
            end: Long,
            network: ComputationGraph,
            portfolioId: String,
            modelId: String,
            conf: Config): Future[ComputationGraph] = {

    val tradingFrequency = conf.getInt("tradr.trader.tradingFrequency")
    val gamma = conf.getDouble("tradr.predictor.a3c.gamma")

    // Step through the time window by steps of length batchsize
    val steps = (start until end by network.batchSize())
      .zip(start + network.batchSize() to end by network.batchSize())

    // Create a future of the computation graph to update it
    val futureNetwork = Future{network}

    // Go through the shuffled steps and update the network
    val updatedNetwork = Random
      .shuffle(steps)
      .foldLeft(futureNetwork){
        case (net, (from, until)) =>
          // Get predicitions of the time frame
          val predictions = CassandraConnector
            .getPredictions(from, until, modelId, conf)
            .map{
              predValSeq => predValSeq.map{
                case (name, timestamp, probabilities, valuePrediction) =>
                  PredictionResult(
                    "predictionID",
                    timestamp,
                    name,
                    Map(
                      "probabilities" -> probabilities,
                      "valuePrediction" -> valuePrediction
                    )
                  )
//                  A3CPredictionResult(timestamp, name, probabilities, valuePrediction)
              }
            }

          // Get portfolio snapshots of the time frame
          val portfolios = CassandraConnector.getPortfolioValues(portfolioId, from, until, conf)

          // Map the predictions and portfolios end each other
          val predictionPortfolioPairs = mapPredictionsAndPortfolios(predictions, portfolios, tradingFrequency)

          // Compute the gradient for each prediction/portfolio pair
          update(model.network, gamma, predictionPortfolioPairs)
      }

    updatedNetwork.map(net => model.copy(network = net))

  }

}






