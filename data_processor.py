import csv
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class AdvisingProcessor(DataProcessor):
    """Processor for the DSTC7 Advising data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples("train", os.path.join(data_dir, "dstc8_train_eo_src.txt"),
                                     os.path.join(data_dir, "dstc8_train_eo_tgt.txt"))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples("dev", os.path.join(data_dir, "dstc8_dev_eo_src.txt"),
                                     os.path.join(data_dir, "dstc8_dev_eo_tgt.txt"))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, set_type, src, tgt):
        """Creates examples for the training and dev sets."""
        examples = []
        fs = open(src, "r", encoding="utf-8")
        ft = open(tgt, "r", encoding="utf-8")

        for s, t in zip(fs.readlines(), ft.readlines()):
            dialog_id, ans_idx, candi_idx, candi_sent = t.split("__DELIM__")
            guid = "%s-%s-%s" % (set_type, dialog_id, candi_idx)
            text_a = s
            text_b = candi_sent
            label = "1" if ans_idx == candi_idx else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class UbuntuProcessor(DataProcessor):
    """Processor for the Ubuntu data set (DSTC7 subtask 1)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ubuntu_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ubuntu_dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ubuntu_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

