package tradr.model


import play.api.libs.json.{Json, Reads, Writes}
import tradr.common.trainer.TrainingResult

object A3cTrainingResult {

  implicit val A3cTrainingResultReads: Reads[A3cTrainingResult] = Json.reads[A3cTrainingResult]
  implicit val A3cTrainingResultWrites: Writes[A3cTrainingResult] = Json.writes[A3cTrainingResult]

}

case class A3cTrainingResult(
                              timestamp: Long,
                              modelid: String,
                              tradeId: String,
                              loss: Double,
                            ) extends TrainingResult {

}
