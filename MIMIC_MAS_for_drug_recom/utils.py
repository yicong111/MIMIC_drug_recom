import os
import json
import random
from openai import OpenAI
from pptree import *
import time 
import numpy as np
import dill
import jsonlines
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from mistralai import Mistral

def read_jsonlines(data_path):
    '''read data from jsonlines file'''
    data = []

    with jsonlines.open(data_path, "r") as f:
        for meta_data in f:
            data.append(meta_data)

    return data

# def multi_test(prauc, ja, f1):
def multi_test(ja, f1):
    result = []
    for _ in range(10):
        data_num = len(ja)
        final_length = int(0.8 * data_num)
        idx_list = list(range(data_num))
        random.shuffle(idx_list)
        idx_list = idx_list[:final_length]
        avg_ja = np.mean([ja[i] for i in idx_list])
        # avg_prauc = np.mean([prauc[i] for i in idx_list]) #本项目不需要
        avg_f1 = np.mean([f1[i] for i in idx_list])
        # result.append([avg_prauc, avg_ja, avg_f1])
        result.append([avg_ja, avg_f1])
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    return mean, std




def multi_test_anova(ja, num_samples=10, sample_ratio=0.8):
    ja = np.array(ja)  
    data_num = len(ja)
    final_length = int(sample_ratio * data_num)
    result = np.empty(num_samples) 
    for i in range(num_samples):
        idx_list = np.random.choice(data_num, final_length, replace=False)  # 随机采样
        result[i] = np.mean(ja[idx_list])  # 计算均值
    return result.mean(), result.std()


def multi_label_metric(y_gt, y_pred, y_prob): #分别是真实值、预测值（0 or 1)、预测概率(0-1)

    def jaccard(y_gt, y_pred): #相似度 = 交集/并集
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return score

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

 
    f1 = f1(y_gt, y_pred)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    # mean, std = multi_test(prauc, ja, avg_f1)
    mean, std = multi_test(ja, avg_f1)
    return np.mean(ja), np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1), mean, std


def multi_label_metric_anova(y_gt, y_pred, y_prob): #分别是真实值、预测值（0 or 1)、预测概率(0-1)
    def jaccard(y_gt, y_pred): #相似度 = 交集/并集
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return score

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score
    
    
    def average_drug_num(y_gt, y_pred):
         # np.sum(y_gt, axis=1).tolist()
        num = []
        for b in range(y_gt.shape[0]):
            out_list = np.where(y_pred[b] == 1)[0]
            drug_num = len(out_list)
            num.append(drug_num)
        return num

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)
    f1 = f1(y_gt, y_pred)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    drug_num = average_drug_num(y_gt, y_pred)
    # mean, std = multi_test(ja, avg_f1)
    return ja, avg_prc, avg_recall, avg_f1, drug_num



def ddi_rate_score(record, ddi_A):
    # ddi rate
    # ddi_A = pickle.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    # print(dd_cnt, all_cnt)
    return dd_cnt / all_cnt

def ddi_pairs(record, ddi_A):
    ddi_pair = []
    all_pair = []
    for patient in record:
        for adm in patient:
            all_cnt = 0
            dd_cnt = 0
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
            ddi_pair.append(dd_cnt)
            all_pair.append(all_cnt)
    return ddi_pair, all_pair

def compute_ddi_rate_stats(ddi_pair, all_pair):
    ddi_rates = []
    for dd, total in zip(ddi_pair, all_pair):
        if total == 0:
            continue  # 跳过药物数量少于2的处方（没有药物对）
        ddi_rates.append(dd / total)
    
    mean_ddi_rate = np.mean(ddi_rates)
    std_ddi_rate = np.std(ddi_rates)

    return ddi_rates, mean_ddi_rate, std_ddi_rate


def ddi_rate_score_anova(record, ddi_A, n_iter=10):
    ddi_rates = []
    ddi_rates_ = []
    for patient in record:
        n_patients = len(patient)
        for index in range(n_patients):
            subset_ = []
            new_patient = patient[:index] + patient[index+1:]
            subset_.append(new_patient)
            rate = ddi_rate_score(subset_, ddi_A)
            ddi_rates_.append(rate)
        print(n_patients)
        for _ in range(n_iter):
            # 打乱患者顺序并取80%的样本
            shuffled_patients = random.sample(patient, k=len(patient))  # 创建打乱后的副本
            n_samples = int(0.8 * n_patients)
            subset = shuffled_patients[:n_samples]
            subset_ = []
            subset_.append(subset)
            # 计算当前子集的DDI rate
            rate = ddi_rate_score(subset_, ddi_A)
            ddi_rates.append(rate)
        
        # 计算均值和标准差
        mean = np.mean(ddi_rates)
        std = np.std(ddi_rates, ddof=1)  # 使用样本标准差（无偏估计）
    
    return ddi_rates_, ddi_rates, mean, std


class Voc(object): #创建和管理词汇表
    '''Define the vocabulary (token) dict'''

    def __init__(self):

        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        '''add vocabulary to dict via a list of words'''
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class EHRTokenizer(object):
    """The tokenization that offers function of converting id and token"""

    def __init__(self, voc_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()  # this is a overall Voc

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.diag_voc, self.med_voc, self.pro_voc = self.read_voc(voc_dir)
        self.vocab.add_sentence(self.med_voc.word2idx.keys())
        self.vocab.add_sentence(self.diag_voc.word2idx.keys())
        self.vocab.add_sentence(self.pro_voc.word2idx.keys())

        self.attri_num = None
        self.hos_num = None
    

    def read_voc(self, voc_dir):

        with open(voc_dir, 'rb') as f:
            
            voc_dict = dill.load(f)
            
        return voc_dict['diag_voc'], voc_dict['med_voc'], voc_dict['pro_voc']


    def add_vocab(self, vocab_file):

        voc = self.vocab
        specific_voc = Voc()

        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])

        return specific_voc


    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids


    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens
    

    def convert_med_tokens_to_ids(self, tokens):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        ids = []
        for i in tokens:
            ids.append(self.med_voc.word2idx[i])
        return ids
    

    def convert_med_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.med_voc.idx2word[i])
        return tokens



class Agent:
    def __init__(self, instruction, role, model_info='Qwen2.5-72B-Instruct-GPTQ-Int4', img_path=None): #examplers=None,
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "system", "content": instruction},
        ]
        
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        """
        处理模型推理请求，添加异常处理和重试机制。
        """
        retry_count = 0  # 重试次数
        success = False  # 记录是否推理成功
        # model_name = "Llama3.1-8B-Instruct"  # 指定模型名
        model_name = "Qwen2.5-72B-Instruct-GPTQ-Int4"
        # model_name = "DeepSeek-R1-Distill-Llama-70B"

        while retry_count < max_retries and not success:
            try:
                # 添加用户输入到对话历史
                self.messages.append({"role": "user", "content": message})
                
                # 模型推理请求
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6,
                )
                
                # 获取模型响应内容
                assistant_response = response.choices[0].message.content
                
                # 添加模型输出到对话历史
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  # 标记推理成功
                return assistant_response  # 返回模型输出

            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  # 打印错误信息
                time.sleep(2)  # 等待2秒后重试，避免连续请求过快

        # 如果达到最大重试次数，返回错误信息
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message
    



class Agent_DeepSeek: #火山引擎/讯飞
    def __init__(self, instruction, role, model_info='ep-20250218184959-5dgh6', img_path=None): 

        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = ""
        openai_api_base = "https://ark.cn-beijing.volces.com/api/v3"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "OK."}
        ]
       
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        """
        处理模型推理请求，添加异常处理和重试机制。
        """
        retry_count = 0  # 重试次数
        success = False  # 记录是否推理成功
        model_name = self.model_info

        while retry_count < max_retries and not success:
            try:
                # 添加用户输入到对话历史
                self.messages.append({"role": "user", "content": message})
                
                # 模型推理请求
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6
                    # timeout=3000  # 设置超时时间为30秒
                )
                
                # 获取模型响应内容
                assistant_response = response.choices[0].message.content
                
                # 添加模型输出到对话历史
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  # 标记推理成功
                return assistant_response  # 返回模型输出

            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  # 打印错误信息
                time.sleep(2)  # 等待2秒后重试，避免连续请求过快

        # 如果达到最大重试次数，返回错误信息
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message



class Agent_Mistral:#Mistral-Large-Instruct-2411-GPTQ #Mistral-Small-24B-Instruct-2501 #是否要改成(and only choose the necessary ones + if) (max_rounds3)
    def __init__(self, instruction, role, model_info='Mistral-Small-24B-Instruct-2501', img_path=None): 
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8002/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "OK."}
        ]
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        """
        处理模型推理请求，添加异常处理和重试机制。
        """
        retry_count = 0  # 重试次数
        success = False  # 记录是否推理成功
        model_name = "Mistral-Small-24B-Instruct-2501"

        while retry_count < max_retries and not success:
            try:
                # 添加用户输入到对话历史
                self.messages.append({"role": "user", "content": message})
                
                # 模型推理请求
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6,
                )
                
                # 获取模型响应内容
                assistant_response = response.choices[0].message.content
                
                # 添加模型输出到对话历史
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  # 标记推理成功
                return assistant_response  # 返回模型输出

            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  # 打印错误信息
                time.sleep(2)  # 等待2秒后重试，避免连续请求过快

        # 如果达到最大重试次数，返回错误信息
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message


class Agent_gpt: #chatgpt
    def __init__(self, instruction, role, model_info='gpt-4o', img_path=None): 

        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = ""
        openai_api_base = "https://api.openai.com/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "OK."}
        ]
       
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        """
        处理模型推理请求，添加异常处理和重试机制。
        """
        retry_count = 0  # 重试次数
        success = False  # 记录是否推理成功
        model_name = self.model_info

        while retry_count < max_retries and not success:
            try:
                # 添加用户输入到对话历史
                self.messages.append({"role": "user", "content": message})
                
                # 模型推理请求
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6
                    # timeout=3000  # 设置超时时间为30秒
                )
                
                # 获取模型响应内容
                assistant_response = response.choices[0].message.content
                
                # 添加模型输出到对话历史
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  # 标记推理成功
                return assistant_response  # 返回模型输出

            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  # 打印错误信息
                time.sleep(2)  # 等待2秒后重试，避免连续请求过快

        # 如果达到最大重试次数，返回错误信息
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message