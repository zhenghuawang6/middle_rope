# import os
# os.environ["http_proxy"] = "127.0.0.1:7890"
# os.environ["https_proxy"] = "127.0.0.1:7890"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# import torch
# from typing import List, Tuple  # for type hints
# import numpy as np  # for manipulating arrays
# import pickle  # for saving the embeddings cache
# import plotly.express as px  # for plots
# import random  # for generating run IDs
# import datasets
# from sklearn.model_selection import train_test_split  # for splitting train & test data
# from beir.retrieval import models
# from beir.datasets.data_loader import GenericDataLoader

# data_path = "../data/nfcorpus"
# train_corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="test")
# test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="train")
# embeddings_model= models.SentenceBERT("facebook/contriever-msmarco")

# #得到query的信息
# query_ids = list(test_queries.keys())
# test_queries = [test_queries[qid] for qid in query_ids]
# query_embeddings = embeddings_model.encode_queries(test_queries, batch_size=256, show_progress_bar=True, convert_to_tensor=False)

# #得到corpus的信息
# corpus_ids = list(test_corpus.keys())
# test_corpus = [test_corpus[cid] for cid in corpus_ids]
# corpus_embeddings = embeddings_model.encode_corpus(test_corpus, batch_size=256, show_progress_bar=True, convert_to_tensor=False)

# import pickle
# pickle.dump(query_embeddings,open(os.path.join(data_path,"queryembedding.pickle"),"wb"))
# pickle.dump(corpus_embeddings,open(os.path.join(data_path,"corpusmbedding.pickle"),"wb"))

# del embeddings_model

# def compute_orthogonal_basis(vecs):
#     if len(vecs) >=2:
#         vecs = np.concatenate(vecs, axis=0)
#     else:
#         vecs = np.array(vecs[0])
#     cov_direction = np.cov(vecs.T)
#     eigVals, eigVects = np.linalg.eigh(cov_direction)
#     eigValsInd=np.argsort(eigVals)[::-1]
#     return eigVals[eigValsInd], (eigVects[:,eigValsInd])
# #得到一个总的
# query_embeddings = query_embeddings
# corpus_embeddings = corpus_embeddings
# vals, vects = compute_orthogonal_basis((query_embeddings, corpus_embeddings))

# import torch.nn as nn
# #加一个各向异性的loss
# class whiteness_adapter(nn.Module):

#     def __init__(self,is_dot = True):
#         super(whiteness_adapter, self).__init__()
#         self.directions = nn.Parameter(torch.tensor(vects).float(), requires_grad=False)
#         self.scales1 = nn.Parameter(torch.ones(100).float(), requires_grad=True)
#         self.scales2 = nn.Parameter(torch.ones(vects.shape[1]-100).float(), requires_grad=False)
#         self.is_dot = is_dot

#     def forward(self, input_embeddings):
#         map_result = torch.matmul(input_embeddings, self.directions)
#         #进行拼接
#         total_scale = torch.cat((self.scales1,self.scales2),dim=-1)
#         scale_result = torch.matmul(map_result, torch.diag(total_scale))
#         iso = torch.std(torch.std(scale_result,dim=0)) #各个维度的一个方差
#         if self.is_dot:
#             return self.dot_sim(scale_result[0:1,:],scale_result)[:,1:], torch.abs(1-self.scales1).sum(), iso
#         else:
#             return self.cos_sim(scale_result[0:1,:],scale_result)[:,1:], torch.abs(1-self.scales1).sum(), iso
    
#     def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
#         a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
#         b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
#         return torch.mm(a_norm, b_norm.transpose(0, 1)) 
    
#     def dot_sim(self, a: torch.Tensor, b: torch.Tensor):
#         return torch.mm(a, b.transpose(0, 1)) 
    
#     #将id与对应的embedding组成字典
# def to_dict(ids, embeddings):
#     result = {}
#     for i, id in enumerate(ids):
#         result[id] = embeddings[i]
#     return result

# query_dict = to_dict(query_ids, query_embeddings)
# corpus_dict = to_dict(corpus_ids, corpus_embeddings)
# #两个字典进行合并
# total_dict = query_dict | corpus_dict

# from datasets import load_dataset
# from torch.utils.data import DataLoader
# dataset = load_dataset("json",data_files="/data/wangzh/bert_white/adapt_whiteness/data/nfcorpus/contriever-msmarco_train_pairwise_qrel.json", split="train")
# dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

# from tqdm import tqdm,trange
# import torch.nn as nn
# import json
# from torch.utils.tensorboard import SummaryWriter   
# import logging
# logging.basicConfig(filename='basic.log',  # logging to a file
#                     level=logging.INFO,   # set level
#                     # change format
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%Y/%m/%d %I:%M:%S %p'
#                     )
# logger = logging.getLogger(__name__)
# for iso_scale in [15,20,25,30,35,40]:
#     logger.info(f"scale为{iso_scale}")
#     writer = SummaryWriter('../log')
#     whiteness_model = whiteness_adapter().cuda().train()
#     op = torch.optim.Adam(whiteness_model.parameters(),lr=0.01)
#     hard_qrels = json.load(open(os.path.join(data_path,"contriever-msmarco_train_qrel.json"),"r"))
#     total_len = len(list(hard_qrels.keys()))
#     i = 0

#     for epoch in range(2):
#         for train_data in tqdm(dataloader):
#             input_data = []
#             input_data.append(total_dict[train_data["query_id"][0]])
#             for doc_data in train_data["positive:"]:
#                 input_data.append(total_dict[doc_data[0]])
#             input_data = np.stack(input_data, axis=0)
#             input_data = torch.tensor(input_data).float().cuda()
#             output, scale_loss, iso_loss = whiteness_model(input_data)
#             #计算rank loss
#             rank_loss = torch.log(1.2+torch.exp(output[:,1:]-output[0][0])).mean()
#             loss = rank_loss + 0.1*scale_loss + iso_scale*iso_loss
#             writer.add_scalar("rankloss",scalar_value=rank_loss,global_step=i)
#             writer.add_scalar("scale_loss",scalar_value=scale_loss,global_step=i)
#             writer.add_scalar("iso_loss",scalar_value=iso_loss,global_step=i)
#             writer.add_scalar("total_loss",scalar_value=loss,global_step=i)
#             loss.backward()
#             nn.utils.clip_grad_norm_(whiteness_model.parameters(), max_norm=1, norm_type=2)
#             op.step()
#             i+=1

#     whiteness_scale = torch.sqrt(1/torch.tensor(vals).float()).cuda()
#     #加载数据
#     data_path = "../data/nfcorpus"
#     # train_corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
#     test_corpus1, test_queries1, test_qrels1 = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev"
#     from beir.retrieval.evaluation import EvaluateRetrieval
#     from beir.retrieval import models
#     from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
#     print(f"训练后scale为{whiteness_model.scales1}")
#     scale = torch.cat((whiteness_model.scales1,whiteness_model.scales2),dim=-1)
#     #提取出相关的direction与scale
#     rescale_matric = whiteness_model.directions @ torch.diag(scale)

#     model = DRES(models.SentenceBERT("facebook/contriever-msmarco",rescale=rescale_matric), batch_size=256)
#     retriever = EvaluateRetrieval(model, score_function="dot")
#     results = retriever.retrieve(test_corpus1, test_queries1)

#     ndcg, _map, recall, precision = retriever.evaluate(test_qrels1, results, retriever.k_values)
#     logger.info(f"指标为{ndcg}")

import time
import threading
import torch
import subprocess

def is_gpu_free(gpu_uuid):
    """检查指定 GPU 是否空闲。"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout.strip().split('\n')
        used_gpus = {line.split(',')[1].strip() for line in output if line}
        return gpu_uuid not in used_gpus
    except Exception as e:
        print(f"Error checking GPU {gpu_uuid} status: {e}")
        return False

def fill_gpu(gpu_id):
    """启动一个线程填满指定 GPU。"""
    print(f"Filling GPU {gpu_id} with dummy workload...")
    try:
        # 设置当前线程的 CUDA 设备
        torch.cuda.set_device(gpu_id)
        # 动态计算需要的张量大小以填满显存
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        available_memory = total_memory * 0.7  # 留出一定空间避免溢出
        tensor_size = int(available_memory // 4)  # 每个 float32 占 4 字节

        # 创建大张量占用显存
        dummy_tensor = torch.empty(tensor_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
        print(f"GPU {gpu_id} filled with tensor of size: {dummy_tensor.nelement()} elements.")

        # 进行简单操作维持占用
        while True:
            dummy_tensor = dummy_tensor * 1.0001
            dummy_tensor = dummy_tensor / 1.0001
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"Stopping GPU filler thread for GPU {gpu_id}...")
    except Exception as e:
        print(f"Error in GPU filler for GPU {gpu_id}: {e}")

def monitor_gpus():
    """监控多张 GPU，并在空闲时启动填充任务。"""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available.")
        return

    gpu_uuids = []
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        gpu_uuids = result.stdout.strip().split('\n')
    except Exception as e:
        print(f"Error fetching GPU UUIDs: {e}")
        return

    filler_threads = [None] * num_gpus
    num_coupy_gpus = 0
    while True:
        if num_coupy_gpus < 2:
            for gpu_id, gpu_uuid in enumerate(gpu_uuids):
                if num_coupy_gpus < 2:
                    if is_gpu_free(gpu_uuid):
                        if filler_threads[gpu_id] is None or not filler_threads[gpu_id].is_alive():
                            filler_threads[gpu_id] = threading.Thread(target=fill_gpu, args=(gpu_id,), daemon=True)
                            filler_threads[gpu_id].start()
                            num_coupy_gpus += 1
                            if num_coupy_gpus == 2:
                                break
        time.sleep(5)
        print("waiting......")
if __name__ == "__main__":
    monitor_gpus()
