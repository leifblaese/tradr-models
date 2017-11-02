package tradr.model.a3c

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j
import play.api.libs.json.{JsValue, Json}
import tradr.cassandraconnector.CassandraConnector
import tradr.common.PricingPoint
import tradr.common.predictor.{PredictionRequest, PredictionResult}
import tradr.common.trading.Instruments

import scala.concurrent.Future

object A3CModel {

  /**
    * Load a network start disk
    * @param id ID of the model that is loaded
    * @param conf Config, has the save file location stored as "tradr.predictor.network.saveFile"
    * @return
    */
  def load(id: String, conf: Config): ComputationGraph = {
    val saveFile: String = conf.getString("tradr.predictor.modelFolder") + id

    println(s"Loading a3c network start $saveFile")
    val f = new File(saveFile)
    if (f.exists()) {
      val network = ModelSerializer.restoreComputationGraph(saveFile)
      network.init()
      network
    } else {
      create(id, conf)
    }
  }

  /**
    * Create a model with a specific id
    * @param id
    * @param conf
    * @return
    */
  def create(id: String, conf: Config): ComputationGraph = {

    val graphConf = getComputationGraph(conf)
    val net = new ComputationGraph(graphConf)
    net.init()
    net
  }

  /**
    * Persist the weights of the model end disk
    * @param network
    * @param conf
    * @param id
    */
  def save(network: ComputationGraph, conf: Config, id: String): Unit = {

    val saveFile: String = conf.getString("tradr.predictor.modelFolder") + id

    ModelSerializer.writeModel(
      network,
      saveFile,
      true
    )
  }


  /**
    * Create a frame for the prediction
    * @param now
    * @param conf
    * @return
    */
  def getFrame(now: Long,
               instrument: Instruments.Value,
               conf: Config): Future[Array[Double]] = {
    val inputSize = conf.getInt("tradr.predictor.frameSize")
    val prev = now - (1000L * 60 * 60)
    val pricingPoints = CassandraConnector.getRates(prev, now, instrument, conf)
    pricingPoints.map{
      points => convertToFrame(points, inputSize, prev, now)
    }
  }

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
    * Answer a prediction request for a certain timestamp
    * @return
    */
  def predict(predictionRequest: String): Map[String, Array[Double]] = {
    val conf = ConfigFactory.load()


    val request = Json.parse(predictionRequest).as[PredictionRequest]
    val modelid = "model1.network"

        Json.toJson(predictionResult)(PredictionResult.writes)

    }



  def predict(
               predictionRequest: PredictionRequest,
               cassandraConnector: CassandraConnector,
               network: ComputationGraph): PredictionResult = {

    val futureFrame = getFrame(request.timestamp, Instruments.get(request.instrument), conf)

    futureFrame.map {
      frame =>
        val network = A3CModel.load(modelid, conf)
        val indFrame = Nd4j.create(frame)
        val indResults = network.output(indFrame)

        val actionAndValuePrediction = indResults
          .map(indarray => indarray.data().asDouble())
          .zipWithIndex
          .map {
            case (arr, 0) => "probabilities" -> arr
            case (arr, 1) => "valueFun" -> arr
          }
          .toMap

        val predictionResult = PredictionResult(
          timestamp = System.currentTimeMillis(),
          modelId = modelid,
          predictionId = 1,
          results = actionAndValuePrediction
        )
    }
  }
}

//case class A3CModel(
//                    id: String,
//                    network: ComputationGraph
//                   ) {
//
//  import A3CModel._
//
//
//}