# pylint: disable=no-member

import json
import os
import pickle
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from deepparse.cli import (
    is_csv_path,
    is_pickle_path,
    to_csv,
    to_pickle,
    generate_export_path,
    is_json_path,
    to_json,
    replace_path_extension,
)
from deepparse.parser import FormattedParsedAddress


class ToolsTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.an_export_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        self.sep = "\t"

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_givenCSVPath_whenCSVPath_returnTrue(self):
        a_csv_path = "a_path.csv"

        self.assertTrue(is_csv_path(a_csv_path))

        a_csv_path = "a/path/a_path.csv"

        self.assertTrue(is_csv_path(a_csv_path))

        a_csv_path = "./relative/path/a_path.csv"

        self.assertTrue(is_csv_path(a_csv_path))

    def test_givenNotACSVPath_whenCSVPath_returnFalse(self):
        not_a_csv_path = "a_path.tsv"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.doc"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.txt"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.p"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.pickle"

        self.assertFalse(is_csv_path(not_a_csv_path))

    def test_givenPicklePath_whenPicklePath_returnTrue(self):
        a_pickle_path = "a_path.pickle"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "a/path/a_path.pickle"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "./relative/path/a_path.pickle"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "a_path.p"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "a/path/a_path.p"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "./relative/path/a_path.p"

        self.assertTrue(is_pickle_path(a_pickle_path))

    def test_givenNotAPicklePath_whenPicklePath_returnFalse(self):
        not_a_pickle_path = "a_path.tsv"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.doc"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.txt"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.csv"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.md"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

    def test_givenJSONPath_whenJSONPath_returnTrue(self):
        a_json_path = "a_path.json"

        self.assertTrue(is_json_path(a_json_path))

        a_json_path = "a/path/a_path.json"

        self.assertTrue(is_json_path(a_json_path))

        a_json_path = "./relative/path/a_path.json"

        self.assertTrue(is_json_path(a_json_path))

    def test_givenNotAJSONPath_whenJSONPath_returnFalse(self):
        not_a_json_path = "a_path.tsv"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.doc"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.txt"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.csv"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.md"

        self.assertFalse(is_json_path(not_a_json_path))

    def test_integration_list_formatted_addresses_to_csv(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_list_of_parsed_addresses = [
            FormattedParsedAddress({a_address_str: a_parsed_address}),
            FormattedParsedAddress({a_address_str: a_parsed_address}),
        ]

        to_csv(a_list_of_parsed_addresses, export_path=self.an_export_path, sep=self.sep)

        parsed_data = pd.read_csv(self.an_export_path, sep=self.sep)
        self.assertEqual(parsed_data.Address[0], a_address_str)
        self.assertEqual(parsed_data.Address[1], a_address_str)

    def test_integration_formatted_address_to_csv(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_parsed_address = FormattedParsedAddress({a_address_str: a_parsed_address})

        to_csv(a_parsed_address, export_path=self.an_export_path, sep=self.sep)

        parsed_data = pd.read_csv(self.an_export_path, sep=self.sep)
        self.assertEqual(parsed_data.Address[0], a_address_str)

    def test_integration_list_formatted_addresses_to_pickle(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_list_of_parsed_addresses = [
            FormattedParsedAddress({a_address_str: a_parsed_address}),
            FormattedParsedAddress({a_address_str: a_parsed_address}),
        ]

        to_pickle(a_list_of_parsed_addresses, export_path=self.an_export_path)

        with open(self.an_export_path, "rb") as file:
            parsed_data = pickle.load(file)
        self.assertEqual(parsed_data[0][0], a_address_str)
        self.assertEqual(parsed_data[1][0], a_address_str)

    def test_integration_formatted_address_to_pickle(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_parsed_address = FormattedParsedAddress({a_address_str: a_parsed_address})

        to_pickle(a_parsed_address, export_path=self.an_export_path)

        with open(self.an_export_path, "rb") as file:
            parsed_data = pickle.load(file)
        self.assertEqual(parsed_data[0][0], a_address_str)

    def test_integration_list_formatted_addresses_to_json(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_list_of_parsed_addresses = [
            FormattedParsedAddress({a_address_str: a_parsed_address}),
            FormattedParsedAddress({a_address_str: a_parsed_address}),
        ]

        to_json(a_list_of_parsed_addresses, export_path=self.an_export_path)

        with open(self.an_export_path, "r", encoding='utf-8') as file:
            parsed_data = json.load(file)
        self.assertIsInstance(parsed_data[0], dict)
        self.assertEqual(parsed_data[0].get("Address"), a_address_str)
        self.assertEqual(parsed_data[1].get("Address"), a_address_str)

    def test_integration_formatted_address_to_json(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_parsed_address = FormattedParsedAddress({a_address_str: a_parsed_address})

        to_json(a_parsed_address, export_path=self.an_export_path)

        with open(self.an_export_path, "r", encoding='utf-8') as file:
            parsed_data = json.load(file)
        self.assertIsInstance(parsed_data[0], dict)
        self.assertEqual(parsed_data[0].get("Address"), a_address_str)

    def test_generate_export_path_export_proper_path(self):
        a_export_file_name = "export.p"

        a_relative_dataset_path = os.path.join(".", "file_name.p")
        actual = generate_export_path(a_relative_dataset_path, a_export_file_name)
        expected = os.path.join(".", a_export_file_name)
        self.assertEqual(actual, expected)

        an_absolute_dataset_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        actual = generate_export_path(an_absolute_dataset_path, a_export_file_name)
        expected = os.path.join(self.temp_dir_obj.name, a_export_file_name)
        self.assertEqual(actual, expected)

    def test_replace_path_extension(self):
        a_relative_dataset_path = os.path.join(".", "file_name.p")
        actual = replace_path_extension(a_relative_dataset_path, ".log")
        expected = a_relative_dataset_path.replace(".p", ".log")

        self.assertEqual(actual, expected)

        an_absolute_dataset_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        actual = replace_path_extension(an_absolute_dataset_path, ".log")
        expected = an_absolute_dataset_path.replace(".p", ".log")

        self.assertEqual(actual, expected)

        an_absolute_dataset_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        actual = replace_path_extension(an_absolute_dataset_path, ".txt")
        expected = an_absolute_dataset_path.replace(".p", ".txt")

        self.assertEqual(actual, expected)
