package tradr.model.a3c

import tradr.common.trading.Action

case class A3CPredictionResult(
                              timestamp: Long,
                              model: String,
                              actionProbabilities: Array[Double],
                              valuePrediction: Double
                              ) {
}
