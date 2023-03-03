from typing import Union, List

class SeqLabelingInputExample:
    """
    Structure for one sequence labeling input example with list of tokens, list of labels and a unique id
    """
    def __init__(self, guid: str, text: List[str], label: List[str]):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the text for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.seq = text
        self.label = label