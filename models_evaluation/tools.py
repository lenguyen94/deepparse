import json
import os
from typing import List

import pandas as pd
import pycountry

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def clean_up_name(country: str) -> str:
    """
    Function to clean up pycountry name
    """
    if "Korea" in country:
        country = "South Korea"
    elif "Russian Federation" in country:
        country = "Russia"
    elif "Venezuela" in country:
        country = "Venezuela"
    elif "Moldova" in country:
        country = "Moldova"
    elif "Bosnia" in country:
        country = "Bosnia"
    return country


# country that we trained on
train_test_files = [
    "br.p", "us.p", "kp.p", "ru.p", "de.p", "fr.p", "nl.p", "ch.p", "fi.p", "es.p", "cz.p", "gb.p", "mx.p", "no.p",
    "ca.p", "it.p", "au.p", "dk.p", "pl.p", "at.p"
]


def train_country_file(file: str) -> bool:
    """
    Validate if a file is a training country (as reference of our article).
    """
    return file in train_test_files


# country that we did not train on
other_test_files = [
    "ie.p", "rs.p", "uz.p", "ua.p", "za.p", "py.p", "gr.p", "dz.p", "by.p", "se.p", "pt.p", "hu.p", "is.p", "co.p",
    "lv.p", "my.p", "ba.p", "in.p", "re.p", "hr.p", "ee.p", "nc.p", "jp.p", "nz.p", "sg.p", "ro.p", "bd.p", "sk.p",
    "ar.p", "kz.p", "ve.p", "id.p", "bg.p", "cy.p", "bm.p", "md.p", "si.p", "lt.p", "ph.p", "be.p", "fo.p"
]


def zero_shot_eval_country_file(file: str) -> bool:
    """
    Validate if a file is a zero shot country (as reference of our article).
    """
    return file in other_test_files


def convert_2_letters_name_into_country_name(country_file_name: str) -> str:
    country_name = pycountry.countries.get(alpha_2=country_file_name.replace(".p", "").upper()).name
    country_name = clean_up_name(country_name)
    return country_name


def convert_two_letter_name_to_country_name_in_json(file_path: str) -> None:
    """
    Function to convert the name of the countries in a results file into their complete name.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    new_data = {}
    for country_name, value in data.items():
        new_data.update({convert_2_letters_name_into_country_name(country_name): value})

    with open(file_path, "w") as file:
        json.dump(new_data, file)


def test_on_country_data(address_parser: AddressParser, file: str, directory_path: str, args) -> tuple:
    """
    Compute the results over a country data.
    """
    country = convert_2_letters_name_into_country_name(file)

    print(f"Testing on test files {country}")

    test_file_path = os.path.join(directory_path, file)
    test_container = PickleDatasetContainer(test_file_path)

    results = address_parser.test(test_container,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  logging_path=f"./checkpoints/{args.model_type}",
                                  checkpoint=args.model_path)
    return results, country


def make_table(data_type: str, root_path: str = "."):
    """
    Function to generate an Markdown table
    """
    table_dir = os.path.join("tables", "actual")
    os.makedirs(table_dir, exist_ok=True)

    fasttext_all_res = json.load(open(os.path.join(root_path, f"{data_type}_test_results_fasttext.json"), "r"))
    bpemb_all_res = json.load(open(os.path.join(root_path, f"{data_type}_test_results_bpemb.json"), "r"))

    formatted_data = []
    # we format the data to have two pairs of columns for a less long table
    for idx, ((country, fasttext_res), (_, bpemb_res)) in enumerate(zip(fasttext_all_res.items(),
                                                                        bpemb_all_res.items())):
        if idx % 2 and idx != 0:
            data.extend([country, fasttext_res, bpemb_res])
            formatted_data.append(data)
        else:
            data = [country, fasttext_res, bpemb_res]
            if idx == 40:
                formatted_data.append(data)
    table = pd.DataFrame(formatted_data,
                         columns=["Country", r"Fasttext (%)", r"BPEmb (%)", "Country", r"Fasttext (%)",
                                  r"BPEmb (%)"]).round(2).to_markdown(index=False)

    with open(os.path.join(table_dir, f"{data_type}_table.md"), "w", encoding="utf-8") as file:
        file.writelines(table)


def make_table_rst(data_type: str, root_path: str = "."):
    # pylint: disable=too-many-locals
    """
    Function to generate an Sphinx RST table
    """
    table_dir = os.path.join("tables", "actual")
    os.makedirs(table_dir, exist_ok=True)

    fasttext_all_res = json.load(open(os.path.join(root_path, f"{data_type}_test_results_fasttext.json"), "r"))
    bpemb_all_res = json.load(open(os.path.join(root_path, f"{data_type}_test_results_bpemb.json"), "r"))

    formatted_data = []
    # we format the data to have two pairs of columns for a less long table
    for idx, ((country, fasttext_res), (_, bpemb_res)) in enumerate(zip(fasttext_all_res.items(),
                                                                        bpemb_all_res.items())):
        if idx % 2 and idx != 0:
            data.extend([country, fasttext_res, bpemb_res])
            formatted_data.append(data)
        else:
            data = [country, fasttext_res, bpemb_res]
            if idx == 40:
                formatted_data.append(data)
    table = pd.DataFrame(formatted_data,
                         columns=["Country", r"Fasttext (%)", r"BPEmb (%)", "Country", r"Fasttext (%)",
                                  r"BPEmb (%)"]).round(2)
    new_line_prefix = "\t\t"
    string = ".. list-table::\n" + new_line_prefix + ":header-rows: 1\n" + "\n"

    for idx, column in enumerate(table.columns):
        if idx == 0:
            string = string + new_line_prefix + "*" + f"\t- {column}\n"
        else:
            string = string + new_line_prefix + f"\t- {column}\n"

    for _, row in table.iterrows():
        for idx, data in enumerate(list(row)):
            if idx == 0:
                string = string + new_line_prefix + "*" + f"\t- {data}\n"
            else:
                string = string + new_line_prefix + f"\t- {data}\n"

    with open(os.path.join(table_dir, f"{data_type}_table.rst"), "w", encoding="utf-8") as file:
        file.writelines(string)


def make_comparison_table(results_files_name: List[str], table_name_suffix: str = "training", zero_shot: bool = False):
    """
    Function to generate an Markdown table
    """
    table_dir = os.path.join("tables", "comparison")
    os.makedirs(table_dir, exist_ok=True)

    models_res = [json.load(open(file_name, "r")) for file_name in results_files_name]

    formatted_data = {
        "Finland": [],
        "Canada": [],
        "South Korea": [],
        "United Kingdom": [],
        "Switzerland": [],
        "Italy": [],
        "Brazil": [],
        "France": [],
        "Denmark": [],
        "Norway": [],
        "Netherlands": [],
        "Germany": [],
        "Russia": [],
        "Mexico": [],
        "Czechia": [],
        "Australia": [],
        "Austria": [],
        "Spain": [],
        "United States": [],
        "Poland": []
    }
    if zero_shot:
        formatted_data = {
            "India": [],
            "Portugal": [],
            "Ireland": [],
            "Uzbekistan": [],
            "Slovakia": [],
            "Japan": [],
            "Faroe Islands": [],
            "Slovenia": [],
            "Hungary": [],
            "Venezuela": [],
            "Bermuda": [],
            "Argentina": [],
            "Colombia": [],
            "Cyprus": [],
            "Belarus": [],
            "Bulgaria": [],
            "Estonia": [],
            "Algeria": [],
            "Philippines": [],
            "Réunion": [],
            "Malaysia": [],
            "New Caledonia": [],
            "Lithuania": [],
            "Bosnia": [],
            "Bangladesh": [],
            "Sweden": [],
            "New Zealand": [],
            "Serbia": [],
            "Latvia": [],
            "Romania": [],
            "Paraguay": [],
            "Belgium": [],
            "Kazakhstan": [],
            "South Africa": [],
            "Indonesia": [],
            "Moldova": [],
            "Iceland": [],
            "Singapore": [],
            "Croatia": [],
            "Greece": [],
            "Ukraine": []
        }
    for model_res in models_res:
        for country, res in model_res.items():
            country_data = formatted_data.get(country)
            country_data.append(res)
    table_data = []
    # we format the data to have two pairs of columns for a less long table
    for idx, (country, res) in enumerate(formatted_data.items()):
        if idx % 2 and idx != 0:
            data.extend([country, *res])
            table_data.append(data)
        else:
            data = [country, *res]
            if idx == 2 * len(formatted_data.items()):
                table_data.append(data)

    model_names = [fr"Model {idx} (%)" for idx in range(len(results_files_name))]
    df = pd.DataFrame(table_data,
                         columns=["Country", *model_names, "Country", *model_names]).round(2)
    table = df.to_markdown(index=False)

    with open(os.path.join(table_dir, f"{table_name_suffix}_comparison_table.md"), "w", encoding="utf-8") as file:
        file.writelines(table)
