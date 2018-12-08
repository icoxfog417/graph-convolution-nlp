import os
import sys
from sklearn.metrics import classification_report
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.classification.trainer import Trainer
from gcn.classification.graph_based_classifier import GraphBasedClassifier


def main():
    root = os.path.join(os.path.dirname(__file__), "../../")
    dataset = MultiNLIDataset(root)
    trainer = Trainer(root, log_dir="classifier")
    trainer.build_similarity_graph_trainer()

    sequence_length = 25
    vocab_size = len(trainer.preprocessor.vocabulary.get())

    def preprocessor(x):
        _x = trainer.preprocess(x, sequence_length)
        values = (_x["text"], _x["graph"])
        return values

    model = GraphBasedClassifier(vocab_size, sequence_length)
    model.build(trainer.num_classes, preprocessor)

    metrics = trainer.train(model.model, epochs=25)

    test_data = dataset.test_data()
    y_pred = model.predict(test_data["text"])

    print(classification_report(test_data["label"], y_pred,
                                target_names=dataset.labels()))


if __name__ == "__main__":
    main()
