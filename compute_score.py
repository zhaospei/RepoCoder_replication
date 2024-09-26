# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import editdistance
from collections import defaultdict

from utils import Tools

def compute_EM(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)

def compute_ES(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)

def compute_score_by_repo_with_metadata(repos, lines, stype, passk=1):
    scores = defaultdict(list)
    for line in lines:
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos:
            continue
        samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]
        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)
        scores[repo].append(score)
    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}
    print(stype)
    for repo in avg_scores.keys():
        print(f'{avg_scores[repo]}\t{repo_count[repo]}\t{repo}')
    sum_scores = 0
    len_scores = 0
    for repo in scores:
        sum_scores += sum(scores[repo])
        len_scores += len(scores[repo])
    print(f'Total {stype} avarage: ', round(sum_scores/ len_scores, 4) * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='rust', type=str)
    parser.add_argument("--prediction_path", default=None, type=str)
    args = parser.parse_args() 
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
    if args.prediction_path == None:
        print('prediction path is NONE!')
    '''compute single prediction'''
    print(f'Starting compute score for {args.prediction_path} in {args.language}')
    compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(args.prediction_path), 'EM', passk=1)
    compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(args.prediction_path), 'ES', passk=1)
