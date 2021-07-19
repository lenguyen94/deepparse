import os

from models_evaluation.tools import make_table, make_table_rst, make_comparison_table

# actual table
root_path = os.path.join(".", "results", "actual")
make_table(data_type="training", root_path=root_path)
make_table(data_type="training_incomplete", root_path=root_path)
make_table(data_type="zero_shot", root_path=root_path)

make_table_rst(data_type="training", root_path=root_path)
make_table_rst(data_type="training_incomplete", root_path=root_path)
make_table_rst(data_type="zero_shot", root_path=root_path)

# comparison table training dataset
root_path = os.path.join(".", "results", "actual")
results_fasttext_file_name = os.path.join(root_path, "training_incomplete_test_results_fasttext.json")
results_bpemb_file_name = os.path.join(root_path, "training_incomplete_test_results_bpemb.json")
root_path = os.path.join(".", "results", "new")
results_transformer_file_name = os.path.join(root_path, "training_incomplete_test_results_transformer.json")
make_comparison_table(
    results_files_name=[results_fasttext_file_name, results_bpemb_file_name, results_transformer_file_name])

# comparison table zero shot
table_name_suffix = "zero_shot"
root_path = os.path.join(".", "results", "actual")
results_fasttext_file_name = os.path.join(root_path, "zero_shot_test_results_fasttext.json")
results_bpemb_file_name = os.path.join(root_path, "zero_shot_test_results_bpemb.json")
root_path = os.path.join(".", "results", "new")
results_transformer_file_name = os.path.join(root_path, "zero_shot_incomplete_test_results_transformer.json")
make_comparison_table(
    results_files_name=[results_fasttext_file_name, results_bpemb_file_name, results_transformer_file_name],
    table_name_suffix=table_name_suffix, zero_shot=True)
