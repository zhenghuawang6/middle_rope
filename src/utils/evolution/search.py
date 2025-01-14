# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import json
import socket
import random
import logging
import argparse
import warnings
import datetime

import torch
import numpy as np
import transformers

sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir))
from evolution.algorithms import Evaluator, DimMonoGeneticAlgorithm


logger = logging.getLogger(__file__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    sock = socket.socket()
    sock.bind(('localhost', 0))
    host, port = sock.getsockname()
    logger.info(f'Initialize server on host={host}, port={port}')

    device_list = list(range(torch.cuda.device_count()))
    evaluators: list[Evaluator] = []
    sock.listen(len(device_list))

    #一张卡上一张推理进程
    for device_id in device_list:
        evaluators.append(Evaluator(
            sock=sock,
            args={
                "input_path" : args.input_path, # 输入路径
                "output_path" : args.output_path, # 输出路径
                "num_doc" : args.num_doc, # 文档数量
                "model_name" : args.model_name, # 模型名称
                "sample_num" : args.sample_num, # 采样的数量  
                "batch_size" : args.batch_size, # 处理的批大小
                'host': host, # 主机ip
                'port': port, # 主机端口
            },
            device_list=[device_id],
        ))
    
    #一旦脚本加载好相关的模型以及数据集，就会发送模型“已经准备好”的信号。
    for evaluator in evaluators:
        evaluator.model_ready()

    logger.info(f"Loading model config: {args.model}")
    set_seed()

    if args.hyper_params is None:
        hyper_params_path = os.path.join(os.path.split(__file__)[0], 'default_hyper_params', f'{args.algorithm}.json')
    else:
        hyper_params_path = args.hyper_params

    logger.info(f'Load hyper-parameters from {hyper_params_path}')
    with open(hyper_params_path) as f:
        hyper_params = json.loads(f.read())

    init_points=[[0,1.5],[10,1.5],[20,1.5],[30,1.5]]

    final_points = DimMonoGeneticAlgorithm(
        evaluators=evaluators,
        hyper_params=hyper_params,
        init_points=init_points,
        log_json_path=os.path.join(args.output_dir, f'log-{args.timestamp}.json'), #日志的输出路径
        output_dir=args.output_dir,
        recovery=args.recovery,
    ).run_genetic_algorithm()

    
    np.savetxt(
        os.path.join(args.output_dir, f"result_final.csv"),
        np.array(final_points),
        delimiter='\n',
    )
    #给所有的进程发送信号，代表评价已经完成
    for evaluator in evaluators:
        evaluator.finalize()
    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--yarn-settings", type=str, choices=["mistral", "llama"], default="llama")
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--target-length", type=int)
    parser.add_argument("--dataset-min-tokens", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--ppl-sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--attn-sliding-window", type=int, default=-1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--hyper-params", type=str, default=None)
    parser.add_argument("--init-factors", type=str, default=None)
    parser.add_argument("--auto-rescale-init-factors", action="store_true")
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--recovery", type=str, default=None)
    parser.add_argument("--save-memory", action="store_true")
    parser.add_argument("--model-size-gb", type=float, default=14)
    parser.add_argument("--devices", type=str, default=None)
    args = parser.parse_args()
    args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    warnings.simplefilter("ignore")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(name)s | %(levelname)s]\n%(message)s\n',
        datefmt='%m-%d %H:%M',
        filename=os.path.join(args.output_dir, f'log-{args.timestamp}.txt'),
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__file__)

    main(args)
