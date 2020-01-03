#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft. All rights reserved.

"""Tests for `task_config.py` package."""
import tempfile
import pytest
import os

from unittest import TestCase
from mt_dnn.config.task_config import (
    TaskConfig,
    CreateTaskConfig,
    SUPPORTED_TASKS_MAP,
    COLATaskConfig,
    MNLITaskConfig,
    MRPCTaskConfig,
    QNLITaskConfig,
    QQPTaskConfig,
    RTETaskConfig,
    SCITAILTaskConfig,
    SNLITaskConfig,
    SSTTaskConfig,
    STSBTaskConfig,
    WNLITaskConfig,
    NERTaskConfig,
    POSTaskConfig,
    CHUNKTaskConfig,
    SQUADTaskConfig,
)


@pytest.fixture(scope="function")
def test_object():
    """Fixture to create test config objects.
    This method sets up a fixture that accepts a request object into a test function
    """
    print("setUp: test squad object creation")
    data = {
        "task_name": "squad",
        "data_format": "PremiseAndOneHypothesis",
        "encoder_type": "BERT",
        "dropout_p": 0.05,
        "enable_san": False,
        "metric_meta": ["ACC", "MCC"],
        "n_class": 1,
        "task_type": "Span",
        "split_names": ["train", "dev"],
    }
    yield data
    print("tearDown: test squad object deletion")


class TaskConfigTests(TestCase):
    def get_base_object(self):
        """ Return basic TaskConfig data
        """
        return {
            "data_format": "PremiseOnly",
            "encoder_type": "BERT",
            "enable_san": False,
            "metric_meta": ["ACC"],
            "n_class": 2,
            "task_type": "Classification",
        }

    def test_save_config_file(self):
        """Test saving created configuration file to disk"""
        kwargs = self.get_base_object()
        kwargs.update({"task_name": "cola", "dropout_p": 0.05})
        config = CreateTaskConfig(**kwargs)
        path = tempfile.TemporaryDirectory()
        config.save_pretrained(save_directory=path.name)
        self.assertTrue(os.path.exists(path.name))

    def test_create_task_config(self):
        """Test the creation of task configs"""
        kwargs = self.get_base_object()
        kwargs.update({"task_name": "cola", "dropout_p": 0.05})
        config = CreateTaskConfig(**kwargs)
        self.assertIsInstance(config.get_configured_task(), COLATaskConfig)

    def test_created_task_name(self):
        """Test the creation of task config name"""
        kwargs = self.get_base_object()
        kwargs.update({"task_name": "cola", "dropout_p": 0.05})
        config = CreateTaskConfig(**kwargs)
        self.assertEqual(config.get_task_name(), "cola")

    def test_cola_task_config(self):
        """Test COLA task creation"""
        kwargs = self.get_base_object()
        kwargs.update({"dropout_p": 0.05})
        config = COLATaskConfig(**kwargs)
        self.assertIsInstance(config, COLATaskConfig)

    def test_mnli_task_config(self):
        """Test MNLI task creation"""
        kwargs = self.get_base_object()
        kwargs.update(
            {
                "dropout_p": 0.3,
                "split_names": [
                    "train",
                    "matched_dev",
                    "mismatched_dev",
                    "matched_test",
                    "mismatched_test",
                ],
            }
        )
        config = MNLITaskConfig(**kwargs)
        self.assertIsInstance(config, MNLITaskConfig)

    def test_mrpc_task_config(self):
        """Test MRPC task creation"""
        kwargs = self.get_base_object()
        config = MRPCTaskConfig(**kwargs)
        self.assertIsInstance(config, MRPCTaskConfig)

    def test_qnli_task_config(self):
        """Test QNLI task creation"""
        kwargs = self.get_base_object()
        kwargs.update({"labels": ["not_entailment", "entailment"]})
        config = QNLITaskConfig(**kwargs)
        self.assertIsInstance(config, QNLITaskConfig)

    def test_qqp_task_config(self):
        """Test QQP task creation"""
        kwargs = self.get_base_object()
        config = QQPTaskConfig(**kwargs)
        self.assertIsInstance(config, QQPTaskConfig)

    def test_rte_task_config(self):
        """Test RTE task creation"""
        kwargs = self.get_base_object()
        kwargs.update({"labels": ["not_entailment", "entailment"]})
        config = RTETaskConfig(**kwargs)
        self.assertIsInstance(config, RTETaskConfig)

    def test_scitail_task_config(self):
        """Test SCITAIL task creation"""
        kwargs = self.get_base_object()
        kwargs.update({"labels": ["neutral", "entails"]})
        config = SCITAILTaskConfig(**kwargs)
        self.assertIsInstance(config, SCITAILTaskConfig)

    def test_snli_task_config(self):
        """Test SNLI task creation"""
        kwargs = self.get_base_object()
        kwargs.update({"labels": ["contradiction", "neutral", "entailment"]})
        config = SNLITaskConfig(**kwargs)
        self.assertIsInstance(config, SNLITaskConfig)

    def test_sst_task_config(self):
        """Test SST task creation"""
        kwargs = self.get_base_object()
        config = SSTTaskConfig(**kwargs)
        self.assertIsInstance(config, SSTTaskConfig)

    def test_stsb_task_config(self):
        """Test STSB task creation"""
        kwargs = self.get_base_object()
        config = STSBTaskConfig(**kwargs)
        self.assertIsInstance(config, STSBTaskConfig)

    def test_wnli_task_config(self):
        """Test WNLI task creation"""
        kwargs = self.get_base_object()
        config = WNLITaskConfig(**kwargs)
        self.assertIsInstance(config, WNLITaskConfig)

    def test_ner_task_config(self):
        """Test NER task creation"""
        kwargs = self.get_base_object()
        kwargs.update(
            {
                "split_names": ["train", "dev", "test"],
                "labels": [
                    "O",
                    "B-MISC",
                    "I-MISC",
                    "B-PER",
                    "I-PER",
                    "B-ORG",
                    "I-ORG",
                    "B-LOC",
                    "I-LOC",
                    "X",
                    "CLS",
                    "SEP",
                ],
            }
        )
        config = NERTaskConfig(**kwargs)
        self.assertIsInstance(config, NERTaskConfig)

    def test_pos_task_config(self):
        """Test POS task config"""
        kwargs = self.get_base_object()
        kwargs.update(
            {
                "split_names": ["train", "dev", "test"],
                "labels": [
                    ",",
                    "\\",
                    ":",
                    ".",
                    "''",
                    '"',
                    "(",
                    ")",
                    "$",
                    "CC",
                    "CD",
                    "DT",
                    "EX",
                    "FW",
                    "IN",
                    "JJ",
                    "JJR",
                    "JJS",
                    "LS",
                    "MD",
                    "NN",
                    "NNP",
                    "NNPS",
                    "NNS",
                    "NN|SYM",
                    "PDT",
                    "POS",
                    "PRP",
                    "PRP$",
                    "RB",
                    "RBR",
                    "RBS",
                    "RP",
                    "SYM",
                    "TO",
                    "UH",
                    "VB",
                    "VBD",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                    "WDT",
                    "WP",
                    "WP$",
                    "WRB",
                    "X",
                    "CLS",
                    "SEP",
                ],
            }
        )
        config = POSTaskConfig(**kwargs)
        self.assertIsInstance(config, POSTaskConfig)

    def test_chunk_task_config(self):
        """Test CHUNK task creation"""
        kwargs = self.get_base_object()
        kwargs.update(
            {
                "split_names": ["train", "dev", "test"],
                "labels": [
                    "B-ADJP",
                    "B-ADVP",
                    "B-CONJP",
                    "B-INTJ",
                    "B-LST",
                    "B-NP",
                    "B-PP",
                    "B-PRT",
                    "B-SBAR",
                    "B-VP",
                    "I-ADJP",
                    "I-ADVP",
                    "I-CONJP",
                    "I-INTJ",
                    "I-LST",
                    "I-NP",
                    "I-PP",
                    "I-SBAR",
                    "I-VP",
                    "O",
                    "X",
                    "CLS",
                    "SEP",
                ],
            }
        )
        config = CHUNKTaskConfig(**kwargs)
        self.assertIsInstance(config, CHUNKTaskConfig)

    def test_squad_task_config(self):
        """Test SQUAD v1/v2 task creation"""
        kwargs = self.get_base_object()
        kwargs.update({"dropout_p": 0.05, "split_names": ["train", "dev"]})
        config = SQUADTaskConfig(**kwargs)
        self.assertIsInstance(config, SQUADTaskConfig)

    def test_supported_task_maps(self):
        """Test the list of supported task"""
        self.assertEqual(
            list(SUPPORTED_TASKS_MAP.keys()),
            [
                "cola",
                "mnli",
                "mrpc",
                "qnli",
                "qqp",
                "rte",
                "scitail",
                "snli",
                "sst",
                "stsb",
                "wnli",
                "ner",
                "pos",
                "chunk",
                "squad",
                "squad-v2",
            ],
        )

