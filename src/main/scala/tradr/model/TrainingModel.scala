package tradr.model

/**
  * A TrainingModel can do parameter updates for a full minibatch. It can be converted to a predictionModel
  */
trait TrainingModel[T] extends Model[T] {


  def convertToPredictionModel(): PredictionModel

}
