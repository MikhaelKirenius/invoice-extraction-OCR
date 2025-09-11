def build_compute_metrics(id2label, metric):
    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=-1)

        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_preds, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics
