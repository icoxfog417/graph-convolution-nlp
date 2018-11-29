import os
import sys
from sklearn.metrics import classification_report
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.classification.baseline import TfidfClassifier


def main():
    root = os.path.join(os.path.dirname(__file__), "../../")
    dataset = MultiNLIDataset(root)
    classifier = TfidfClassifier()

    train_data = dataset.train_data()
    scores = classifier.fit(train_data["text"], train_data["label"])

    test_data = dataset.test_data()
    y_pred = classifier.model.predict(test_data["text"])

    print(classification_report(test_data["label"], y_pred,
                                target_names=dataset.labels()))


if __name__ == "__main__":
    main()
