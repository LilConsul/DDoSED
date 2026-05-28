import logging

from src.baseline import predict_rule_based
from src.load_dataset import load_dataset
from src.paths import DATASET_PATH
from src.preprocessing import prepare_model_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting DDoS detection pipeline")

    logger.info("Loading dataset from %s", DATASET_PATH)
    dataset = load_dataset(DATASET_PATH)
    logger.info("Dataset loaded: %d rows, %d columns", len(dataset), len(dataset.columns))

    logger.info("Preparing model frame")
    model_frame = prepare_model_frame(dataset)
    logger.info("Model frame prepared: %d rows, %d features", len(model_frame), len(model_frame.columns) - 1)

    logger.info("Running baseline predictions")
    predictions = predict_rule_based(model_frame)

    logger.info("Prediction distribution:")
    for label, count in predictions.value_counts().items():
        percentage = (count / len(predictions)) * 100
        logger.info("  %s: %d (%.2f%%)", label, count, percentage)

    logger.info("Comparing with ground truth")
    accuracy = (predictions == model_frame["target"]).mean()
    logger.info("Baseline accuracy: %.2f%%", accuracy * 100)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
