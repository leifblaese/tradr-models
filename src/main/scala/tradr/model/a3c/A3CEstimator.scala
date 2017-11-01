package tradr.model.a3c

import java.io.File
import java.net.InetSocketAddress
import java.util

import com.datastax.oss.driver.api.core.{Cluster, CqlIdentifier}
import com.datastax.oss.driver.internal.core.config.typesafe.DefaultDriverConfigLoader
import com.typesafe.config.Config
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.gradient.{DefaultGradient, Gradient}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import tradr.cassandraconnector.CassandraConnector
import tradr.common.PricingPoint
import tradr.common.trading._

import scala.collection.mutable
import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}
import scala.util.Random




object A3CEstimator {

  /**
    * Convert the data start cassandra into a (multidimensional) frame
    * We look at a certain time window in cassandra and compute a fixed set of input
    * pixels for the NN. In this very first version, we will just take the mean of pixels
    * within a distinct bin.
    * I.e.: Bin the data, compute Mean of PricingPoints and return as Array
    *
    * If we do not have enough data end fill the array we need end throw an error
    */
  private def convertToFrame(pricingPoints: Seq[PricingPoint],
                             inputSize: Int,
                             start: Long, end: Long): Array[Double] = {

    val stepSize = (end - start)/inputSize
    val range = start until end by stepSize

    range
      .indices
      .map(i => {
        val filteredSet = pricingPoints
          .filter(point => point.timestamp > range(i) && point.timestamp <= range(i+1))
          .map(_.value)
        assert(filteredSet.nonEmpty)
        filteredSet.sum / filteredSet.size.toDouble
      })
      .toArray
  }


  /**
    * Create a frame for the prediction
    * @param now
    * @param conf
    * @return
    */
  def getFrame(now: Long, conf: Config): Future[Array[Double]] = {
    val inputSize = conf.getInt("tradr.predictor.frameSize")
    val prev = now - (1000L * 60 * 60)
    val pricingPoints = CassandraConnector.getRates(prev, now, Instruments.EURUSD, conf)
    pricingPoints.map{
      points => convertToFrame(points, inputSize, prev, now)
    }
  }



  /**
    * Compute the errors for a prediction and the resulting portfolio
    * @param prediction
    * @param portfolio
    * @param initialR
    * @param gamma
    * @param profit
    * @return
    */
  private def computeError(prediction: A3CPredictionResult,
                   portfolio: Portfolio,
                   initialR: Double,
                   gamma: Double,
                   profit: Double) = {

    val R = gamma * initialR + profit

    val actionProbError = prediction
      .actionProbabilities
      .map(Math.log)
      .map(_ * (R - prediction.valuePrediction))

    val valueFunError = Math.pow(R - prediction.valuePrediction, 2.0)

    (actionProbError, valueFunError, R)
  }


  /**
    * Compute errors for a certain time frame
    * @param gamma
    * @param predictionPortfolioPairs
    * @return
    */
  private def computeAllErrors(gamma: Double,
                       predictionPortfolioPairs: Seq[(A3CPredictionResult, Portfolio)]) = {
    val lastCurrencyValue = predictionPortfolioPairs.last._2.currencies(Currencies.EUR)
    val firstR = predictionPortfolioPairs.last._1.valuePrediction


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
              predictionPortfolioPairs: Future[Seq[(A3CPredictionResult, Portfolio)]],
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
    * @param tradingFrequency
    * @return
    */
  def mapPredictionsAndPortfolios(
                                  predictionsFuture: Future[Seq[A3CPredictionResult]],
                                  portfoliosFuture: Future[Seq[(Long, Portfolio)]],
                                  tradingFrequency: Int): Future[Seq[(A3CPredictionResult, Portfolio)]] = {
    implicit val ec: ExecutionContextExecutor = ExecutionContext.global
    predictionsFuture.flatMap {
      predictions =>

        val a: Seq[Future[(A3CPredictionResult, Portfolio)]] = predictions
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
    * @param start
    * @param end
    */
  def train(start: Long,
            end: Long,
            model: A3CModel,
            portfolioId: String,
            conf: Config): Future[A3CModel] = {

    val tradingFrequency = conf.getInt("tradr.trader.tradingFrequency")
    val gamma = conf.getDouble("tradr.predictor.a3c.gamma")

    // Step through the time window by steps of length batchsize
    val steps = (start until end by model.network.batchSize())
      .zip(start + model.network.batchSize() to end by model.network.batchSize())

    // Create a future of the computation graph to update it
    val futureNetwork = Future{model.network}

    // Go through the shuffled steps and update the network
    val updatedNetwork = Random
      .shuffle(steps)
      .foldLeft(futureNetwork){
        case (net, (from, until)) =>
          // Get predicitions of the time frame
          val predictions = CassandraConnector
            .getPredictions(from, until, model.id, conf)
            .map{
              predValSeq => predValSeq.map{
                case (name, timestamp, probabilities, valuePrediction) =>
                  A3CPredictionResult(timestamp, name, probabilities, valuePrediction)
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






