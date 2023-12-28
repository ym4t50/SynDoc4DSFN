import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSVS = {
    # paths from `src` directory
    "SynDocDS": DatasetCSV(
        train="../datasets/csv/SynDocDS/train.csv",
        val="../datasets/csv/SynDocDS/val.csv",
        test="../datasets/csv/SynDocDS/test.csv",  # accually not used.
    ),
    "SDSRD": DatasetCSV(
        train="../datasets/csv/SDSRD/train.csv",
        val="../datasets/csv/SDSRD/test.csv",
        test=None,
    ),
    "Jung": DatasetCSV(
        train="../datasets/csv/Jung/train.csv",
        val="../datasets/csv/Jung/val.csv",
        test="../datasets/csv/Jung/test.csv",
    ),
    "OSR": DatasetCSV(
        train="../datasets/csv/OSR/train.csv",
        val="../datasets/csv/OSR/val.csv",
        test="../datasets/csv/OSR/test.csv",
    ),
    "Kligler": DatasetCSV(
        train="../datasets/csv/Kligler/train.csv",
        val="../datasets/csv/Kligler/val.csv",
        test="../datasets/csv/Kligler/test.csv",
    ),
    "JungAll": DatasetCSV(
        train=None,
        val=None,
        test="../datasets/csv/Jung/all.csv",
    ),
    "OSRAll": DatasetCSV(
        train=None,
        val=None,
        test="../datasets/csv/OSR/all.csv",
    ),
    "KliglerAll": DatasetCSV(
        train=None,
        val=None,
        test="../datasets/csv/Kligler/all.csv",
    ),
}
