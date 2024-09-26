# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper

from utils import CONSTANTS, CodexTokenizer, CodeGenTokenizer

def make_repo_window(repos, window_sizes, slice_sizes, language):
    worker = MakeWindowWrapper(None, repos, window_sizes, slice_sizes, language)
    worker.window_for_repo_files()
    vectorizer = BagOfWords
    BuildVectorWrapper(None, vectorizer, repos, window_sizes, slice_sizes).vectorize_repo_windows()

def run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes):
    # build code snippets for vanilla retrieval-augmented approach and ground truth
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_baseline_and_ground()
    # build vector for vanilla retrieval-augmented approach and ground truth
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_baseline_and_ground_windows()
    # search code for vanilla retrieval-augmented approach and ground truth
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()
    # build prompt for vanilla retrieval-augmented approach and ground truth
    tokenizer = CodeGenTokenizer
    mode = CONSTANTS.rg
    # output_file_path = 'prompts/rg-one-gram-ws-20-ss-2.jsonl'
    print(f'building prompt for {mode}')
    for window_size, slice_size in zip(window_sizes, slice_sizes):
        output_file_path = f'prompts/{benchmark}-rg-one-gram-ws-{window_size}-ss-{slice_size}.jsonl'
        BuildPromptWrapper('one-gram', benchmark, repos, window_size, slice_size, tokenizer).build_first_search_prompt(mode, output_file_path)

    mode = CONSTANTS.gt
    print(f'building prompt for {mode}')
    for window_size, slice_size in zip(window_sizes, slice_sizes):
        output_file_path = f'prompts/{benchmark}-gt-one-gram-ws-{window_size}-ss-{slice_size}.jsonl'
        BuildPromptWrapper('one-gram', benchmark, repos, window_size, slice_size, tokenizer).build_first_search_prompt(mode, output_file_path)

# def run_oracle_method(benchmark, repos, window_sizes, slice_sizes):
#     # build code snippets for vanilla retrieval-augmented approach and ground truth
#     MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_baseline_and_ground()
#     # build vector for vanilla retrieval-augmented approach and ground truth
#     vectorizer = BagOfWords
#     BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_baseline_and_ground_windows()
#     # search code for vanilla retrieval-augmented approach and ground truth
#     CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()
#     # build prompt for vanilla retrieval-augmented approach and ground truth
#     tokenizer = CodeGenTokenizer


def run_RepoCoder_method(mode, benchmark, repos, window_sizes, slice_sizes, prediction_path):
    # mode = CONSTANTS.rgrg
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    tokenizer = CodeGenTokenizer
    for window_size, slice_size in zip(window_sizes, slice_sizes):
        output_file_path = f'prompts/{benchmark}-repocoder-one-gram-ws-{window_size}-ss-{slice_size}-{mode}.jsonl'
        BuildPromptWrapper('one-gram', benchmark, repos, window_size, slice_size, tokenizer).build_prediction_prompt(mode, prediction_path, output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default='rust_line', type=str)
    parser.add_argument("--language", default='rust', type=str)
    parser.add_argument("--mode", default='repo', type=str)
    parser.add_argument("--prediction_path", default=None, type=str)
    parser.add_argument("--window_sizes", nargs='+', type=int, default=[20], help='A list of window_size')
    parser.add_argument("--slice_sizes", nargs='+', type=int, default=[2], help='A list of slice_size')
    args = parser.parse_args()
    print(f"""\
Starting building prompt for:
- benchmark: {args.benchmark}
- language: {args.language}
- mode: {args.mode}
- window_sizes: {args.window_sizes}
- slice_sizes: {args.slice_sizes}
""")
    if args.language == 'python':
        repos = [
            'huggingface_diffusers',
            'nerfstudio-project_nerfstudio',
            'awslabs_fortuna',
            'huggingface_evaluate',
            'google_vizier',
            'alibaba_FederatedScope',
            'pytorch_rl',
            'opendilab_ACE',
        ]
    elif args.language == 'rust':
        repos = [
            'sxyazi_yazi',
            'ratatui_ratatui',
            'lapce_floem',
            'Julien-cpsn_ATAC',
            'iggy-rs_iggy',
            'Cormanz_smartgpt',
            'lapce_lapdev',
            'spaceandtimelabs_sxt-proof-of-sql',
            'restatedev_restate',
            'FractalFir_rustc_codegen_clr',
        ]

    if args.mode == 'repo':
        # build window for the repos
        make_repo_window(repos, args.window_sizes, args.slice_sizes, args.language)
    elif args.mode == 'r-g-gt':
        # build prompt for the RG1 methods
        run_RG1_and_oracle_method(args.benchmark, repos, args.window_sizes, args.slice_sizes)
    else:
        # build prompt for the RepoCoder method
        if args.prediction_path == None:
            print("You must add prediction_path to run repocoder method")
        else:
            run_RepoCoder_method(args.mode, args.benchmark, repos, args.window_sizes, args.slice_sizes, args.prediction_path)