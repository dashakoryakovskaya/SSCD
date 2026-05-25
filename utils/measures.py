from sklearn.metrics import classification_report, mean_absolute_error
import numpy as np


def mf1(targets: list[np.ndarray] | np.ndarray,
                         predicts: list[np.ndarray] | np.ndarray,
                         return_scores: bool = False) -> float | tuple[float, list[float]]:
    """
    Средняя взвешенная F1-мера
    """
    targets = np.array(targets)
    predicts = np.array(predicts)

    f1_macro_scores = []
    for i in range(predicts.shape[1]):
        cr = classification_report(targets[:, i], predicts[:, i],
                                         output_dict=True, zero_division=0)
        f1_macro_scores.append(cr['macro avg']['f1-score'])

    if return_scores:
        return np.mean(f1_macro_scores), f1_macro_scores
    return np.mean(f1_macro_scores)


def uar(targets: list[np.ndarray] | np.ndarray,
                    predicts: list[np.ndarray] | np.ndarray,
                    return_scores: bool = False) -> float | tuple[float, list[float]]:
    """
    Средняя невзвешенная точность
    """
    targets = np.array(targets)
    predicts = np.array(predicts)

    uar_scores = []
    for i in range(predicts.shape[1]):
        cr = classification_report(targets[:, i], predicts[:, i],
                                         output_dict=True, zero_division=0)
        uar_scores.append(cr['macro avg']['recall'])

    if return_scores:
        return np.mean(uar_scores), uar_scores
    return np.mean(uar_scores)


def acc_func(trues, preds):
    """
    Точность
    """
    acc = []
    for i in range(5):
        acc.append(mean_absolute_error(trues[:, i], preds[:, i]))
    acc = 1 - np.asarray(acc)
    return np.mean(acc)


def ccc(y_true, y_pred):
    """
    Коэффициент корреляции конкордации
    """
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    y_true_var = np.mean(np.square(y_true-y_true_mean))
    y_pred_var = np.mean(np.square(y_pred-y_pred_mean))

    cov = np.mean((y_true-y_true_mean)*(y_pred-y_pred_mean))

    ccc = np.multiply(2., cov) / (y_true_var + y_pred_var + np.square(y_true_mean - y_pred_mean))
    ccc_loss=np.mean(ccc)
    return ccc_loss
