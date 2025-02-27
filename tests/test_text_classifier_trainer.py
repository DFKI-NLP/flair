import shutil

from flairrelex.data import Sentence

from flairrelex.data_fetcher import NLPTaskDataFetcher, NLPTask
from flairrelex.embeddings import WordEmbeddings, DocumentMeanEmbeddings, DocumentLSTMEmbeddings
from flairrelex.models.text_classification_model import TextClassifier
from flairrelex.trainers.text_classification_trainer import TextClassifierTrainer


def test_text_classifier_single_label():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings('en-glove')
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64, False, False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert(l.name is not None)
            assert(0.0 <= l.confidence <= 1.0)
            assert(type(l.confidence) is float)

    # clean up results directory
    shutil.rmtree('./results')


def test_text_classifier_mulit_label():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings('en-glove')
    document_embeddings: DocumentMeanEmbeddings = DocumentMeanEmbeddings([glove_embedding])

    model = TextClassifier(document_embeddings, label_dict, True)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert(l.name is not None)
            assert(0.0 <= l.confidence <= 1.0)
            assert(type(l.confidence) is float)

    # clean up results directory
    shutil.rmtree('./results')