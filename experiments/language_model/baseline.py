import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from gcn.language_model.trainer import Trainer
from gcn.language_model.baseline import LSTMLM


def main():
    root = os.path.join(os.path.dirname(__file__), "../../")
    trainer = Trainer(root, preprocessor_name="baseline_preprocessor",
                      log_dir="baseline")
    trainer.build()
    vocab_size = len(trainer.preprocessor.vocabulary.get())
    print("vocab size: {}".format(vocab_size))
    model = LSTMLM(vocab_size, embedding_size=200, hidden_size=200, layers=2)
    metrics = trainer.train(model, lr=15, decay=0.5, epochs=10)


if __name__ == "__main__":
    main()
