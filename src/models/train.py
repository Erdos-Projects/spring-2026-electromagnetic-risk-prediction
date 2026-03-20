from statistics import mean

import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from src.features.preprocessing import SolarWindPreprocessor
from src.data.fetch_nasa_omni_historical import merge_yearly_dataframes
from src.features.transformers import StormEventExtractor

def preprocess_temporal_splits(dataframes, target_column_label):
    datasplits = []
    preprocessor = SolarWindPreprocessor()
    preprocessor.TARGET = target_column_label

    for k in range(len(dataframes) - 1):
        # Combine training data from splits 0 to k
        train_data =  merge_yearly_dataframes(dataframes[:k+1])
        X_train, y_train = preprocessor.preprocess(train_data, fit_scaler=True, handle_missing='interpolate', remove_outliers_method='iqr', scale=True)

        # Test data is split k+1
        test_data = dataframes[k+1]
        X_test, y_test = preprocessor.preprocess(test_data, fit_scaler=False, handle_missing='interpolate', remove_outliers_method='iqr', scale=True)
    
        datasplits.append((X_train, y_train, X_test, y_test))

    return datasplits

def train_on_temporal_splits(model_class, params, Xs, ys):
    models = []

    for k in range(len(Xs) - 1):
        # Combine training data from splits 0 to k
        X_train =  merge_yearly_dataframes(Xs[:k+1])
        y_train = merge_yearly_dataframes(ys[:k+1])
        
        # Train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        models.append(model)

    return models

def test_on_temporal_splits(models, Xs, ys, probability_threshold=None):
    results = []
    for k in range(len(Xs) - 1):
        # Test data is split k+1
        X_test = Xs[k+1]
        y_test = ys[k+1]
        model = models[k]
        
        # Evaluate on test split
        if not probability_threshold:
            predictions = model.predict(X_test)
        else:
            predictions = (model.predict_proba(X_test)[:, 1] >= probability_threshold).astype(int)

        # Store results
        results.append((predictions, y_test)) 

    return results

def train_and_test_on_temporal_splits(model_class, params, Xs, ys, probability_threshold=None):
    """
    Train and evaluate a model on temporally split data.
    
    For each split k, trains the model on data from splits 0 to k,
    then evaluates on split k+1.
    
    Args:
        model: ML model with fit() and predict() methods
        dataframes: List of dataframes in temporal order
        
    Returns:
        List of evaluation results for each temporal split
    """
    models = train_on_temporal_splits(model_class, params, Xs, ys) 
    return test_on_temporal_splits(models, Xs, ys, probability_threshold=probability_threshold)

def evaluate_storm_predictions(y_true, y_pred):
    """
    Evaluate storm-level prediction quality by comparing contiguous storm events
    extracted from ground-truth and predicted boolean labels.

    A "storm" is any maximal contiguous block of True values (Kp >= threshold).
    Two events are considered a match if their time intervals overlap.

    Args:
        y_true: pd.Series (bool, DatetimeIndex) — ground-truth storm labels.
        y_pred: array-like or pd.Series — predicted storm labels (same length/index as y_true).

    Returns:
        dict with keys:
            storms_accurately_predicted  — list of actual Event objects that were hit
            storms_missed                — list of actual Event objects with no predicted overlap
            false_positives              — list of predicted Event objects with no actual overlap
            storm_recall                 — fraction of actual storms that were hit (or None)
            storm_precision              — fraction of predicted storms that matched an actual (or None)
    """
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index)

    actual_events = StormEventExtractor.extract_true_storm_events(y_true.astype(bool))
    predicted_events = StormEventExtractor.extract_true_storm_events(y_pred.astype(bool))

    def _overlaps(e1, e2):
        return e1.start <= e2.end and e2.start <= e1.end

    hits, missed = [], []
    for actual in actual_events:
        if any(_overlaps(actual, pred) for pred in predicted_events):
            hits.append(actual)
        else:
            missed.append(actual)

    false_positives = [
        pred for pred in predicted_events
        if not any(_overlaps(pred, actual) for actual in actual_events)
    ]

    n_actual = len(actual_events)
    n_pred = len(predicted_events)
    storm_recall = len(hits) / n_actual if n_actual > 0 else None
    storm_precision = len(hits) / n_pred if n_pred > 0 else None

    result = {
        "storm_recall": storm_recall,
        "storm_precision": storm_precision,
        "storms_accurately_predicted": len(hits),
        "storms_missed": len(missed),
        "false_positives": len(false_positives),
    }

    return result

def evaluate_on_temporal_splits(temporal_cross_val_results):
    """
    Evaluate a model on temporally split data without retraining.
    """
    results = temporal_cross_val_results

    fold_recalls = []
    fold_precisions = []
    fold_f1s = []
    storm_recalls = []
    storm_precisions = []
    storms_accurately_predicted = []
    storms_missed = []
    false_positives = []
    for i, split in enumerate(results):
        (predictions, y_val) = split
        fold_recalls.append(recall_score(y_val, predictions, zero_division=0))
        fold_precisions.append(precision_score(y_val, predictions, zero_division=0))
        fold_f1s.append(f1_score(y_val, predictions, zero_division=0))
        storm_results = evaluate_storm_predictions(y_val, predictions)
        storm_recalls.append(storm_results["storm_recall"])
        storm_precisions.append(storm_results["storm_precision"])
        storms_accurately_predicted.append(storm_results["storms_accurately_predicted"])
        storms_missed.append(storm_results["storms_missed"])
        false_positives.append(storm_results["false_positives"])
    return {"Mean recalls": safe_mean(fold_recalls),
            "Mean precisions": safe_mean(fold_precisions),
            "Mean F1 scores": safe_mean(fold_f1s),
            "Mean storm recalls": safe_mean(storm_recalls),
            "Mean storm precisions": safe_mean(storm_precisions),
            "Total storms accurately predicted": sum(storms_accurately_predicted),
            "Total storms missed": sum(storms_missed),
            "Total false positives": sum(false_positives)}

def safe_mean(scores):
    valid_scores = [s for s in scores if s is not None]
    return mean(valid_scores) if valid_scores else None

def time_weighted_mean(scores, num_folds):
    """
    Simple time-weighted mean.
    """
    return sum(scores[i]*(i+1) for i in range(len(scores)) if scores[i])/sum(range(1, num_folds+1))

def class_weight_across_folds(ys):
    y0s = sum([y[y == 0].count() for y in ys])
    y1s = sum([y[y == 1].count() for y in ys])
    return y0s / y1s if y1s > 0 else None