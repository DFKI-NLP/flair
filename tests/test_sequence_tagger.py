import os

import pytest

from flairrelex.data import Sentence
from flairrelex.models import SequenceTagger

@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_tag_sentence():

    # test tagging
    sentence = Sentence('I love Berlin')

    tagger = SequenceTagger.load('ner')

    tagger.predict(sentence)

    # test re-tagging
    tagger = SequenceTagger.load('pos')

    tagger.predict(sentence)
