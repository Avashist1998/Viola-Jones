import argparse
from typing import Dict, Final, List
from os import path, mkdir
import numpy as np
from json import dump

from utils.ada_boost import AdaBoost
from utils.logger import init_logger

Logger: Final = init_logger(__name__)

def get_pred_metrics(y: np.ndarray, pred: np.ndarray):
    """Get the classification metrics."""
    TP = np.sum(np.multiply(((pred) == y).astype("int"), (y == 1).astype("int")))
    TN = np.sum(np.multiply(((pred) == y).astype("int"), (y == -1).astype("int")))
    FP = np.sum(np.multiply(((pred) != y).astype("int"), (y == 1).astype("int")))
    FN = np.sum(np.multiply(((pred) != y).astype("int"), (y == -1).astype("int")))
    return(TP, FP, TN, FN)

def load_file(file_path: str) -> np.ndarray:
    return np.loadtxt(file_path, dtype="int", delimiter=",")

def test_model(dataset_path: str, clf: AdaBoost) -> Dict[int, Dict[str, float]]:
    """Tests the model on a given dataset"""
    test_res: Dict[int, Dict[str: float]] = {}
    Logger.info("Loading testing dataset")
    train_matrix = load_file(dataset_path)
    X, y = train_matrix[:, :-1], train_matrix[:, -1]

    Logger.info("Testing the model ...")
    for i in range(1, clf.num_of_est+1):
        Logger.info(f"Testing the model for round: {i} is starting")
        i_clf_prediction = clf.predict(X, num_of_est=i)
        TP, FP, TN, FN = get_pred_metrics(y, i_clf_prediction)
        i_accuracy = (TP + TN) /(TP+FP+TN+FN)
        i_precision = TP/(TP + FP)
        i_recall = TP/(TP + FN)
        try:
            i_specificity = (TN) /(TN + FP)
        except Exception as e:
            Logger.warning(f" round : {i} has not false positive or true negatives")
            i_specificity = 0
        i_f_score = (2*i_precision*i_recall)/(i_precision + i_recall)
        test_res[i] = {"accuracy": i_accuracy, "precision": i_precision,
                       "recall": i_recall, "specificity": i_specificity, "f_score": i_f_score}
        Logger.info(f"Testing the model for round: {i+1} is complete.")
    return test_res


def save_test_res(test_res: Dict[int, Dict[str, float]], test_file_path: str):
    with open(test_file_path, "w") as fp:
        dump(test_res, fp)

def test(dataset_path: str, model_path: str, output_path: str) -> None:
    
    if not path.exists(output_path):
        Logger.info("Creating output directory")
        mkdir(output_path)

    TEST_PATH: Final = path.join(dataset_path, "test_data.csv")

    clf: AdaBoost = AdaBoost.load(model_path)
    test_res = test_model(TEST_PATH, clf)
    TEST_RESULT_PATH: Final = path.join(output_path, f"{clf.num_of_est}_round_model_test_results.json")
    Logger.info("Saving test results")
    save_test_res(test_res, TEST_RESULT_PATH)
    Logger.info(f"Testing is complete, and results are available at {TEST_RESULT_PATH}")

def main():
    parser = argparse.ArgumentParser(
        prog = "Model Tester",
        description= "Test a given model the generated dataset"
    )
    parser.add_argument("--dataset_path", "-d", type=str)
    parser.add_argument("--model_path", "-m", type=str)
    parser.add_argument("--output_path", "-o", type=str)
    args = parser.parse_args()
    test(args.dataset_path, args.model_path, args.output_path)


if __name__ == "__main__":
    main()
