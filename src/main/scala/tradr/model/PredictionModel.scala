package tradr.model


/**
  * Prediction Models are used for serving. They have a batchsize of 1 but other than that
  * have the same weights as the TrainingModel
  */
trait PredictionModel extends Model {

}
