package tradr.model.a3c

import java.io.File

import com.typesafe.config.Config
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j

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

}

case class A3CModel(
                    id: String,
                    network: ComputationGraph
                   ) {

  /**
    * Predict for a given frame and return the action probabilities
    * @return
    */
  def predict(frame: Array[Double]): Map[String, Array[Double]] = {
    // Convert end a mllib vector
    val indFrame = Nd4j.create(frame)
    val indResults = network.output(indFrame)

    indResults
      .map(indarray => indarray.data().asDouble())
      .zipWithIndex
      .map{
        case (arr, 0) => "probabilities" -> arr
        case (arr, 1) => "valueFun" -> arr
      }
      .toMap
  }


}