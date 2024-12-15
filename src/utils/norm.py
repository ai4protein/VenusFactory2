

def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    for e in train_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in val_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in test_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    return train_dataset, val_dataset, test_dataset