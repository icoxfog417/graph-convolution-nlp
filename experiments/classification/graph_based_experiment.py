import os
import sys
from sklearn.metrics import classification_report
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.classification.trainer import Trainer
from gcn.graph.dependency_graph import DependencyGraph
from gcn.graph.similarity_graph import SimilarityGraph
from gcn.graph.static_graph import StaticGraph
from gcn.classification.graph_based_classifier import GraphBasedClassifier


def main(graph_type="dependency", epochs=25):
    root = os.path.join(os.path.dirname(__file__), "../../")
    dataset = MultiNLIDataset(root)

    if graph_type == "dependency":
        graph_builder = DependencyGraph(lang="en")
    elif graph_type == "similarity":
        graph_builder = SimilarityGraph(lang="en")
    else:
        graph_builder = StaticGraph(lang="en")

    trainer = Trainer(graph_builder, root, log_dir="classifier")
    trainer.build()

    sequence_length = 25
    vocab_size = len(trainer.preprocessor.vocabulary.get())

    def preprocessor(x):
        _x = trainer.preprocess(x, sequence_length)
        values = (_x["text"], _x["graph"])
        return values

    model = GraphBasedClassifier(vocab_size, sequence_length,
                                 lstm=None)
    model.build(trainer.num_classes, preprocessor)

    metrics = trainer.train(model.model, epochs=epochs)

    test_data = dataset.test_data()
    y_pred = model.predict(test_data["text"])

    print(classification_report(test_data["label"], y_pred,
                                target_names=dataset.labels()))


if __name__ == "__main__":
    main()
