import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import pprint as pp
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import json
import faiss
# device = torch.device("cuda:0") 
import numpy as np
from utils import Agent_gpt
import re
import ast
import pandas as pd
import time
from collections import defaultdict
from utils import read_jsonlines, multi_label_metric, ddi_rate_score, Voc, EHRTokenizer
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import random


query_model_name = "ncbi/MedCPT-Query-Encoder"
query_model = AutoModel.from_pretrained(query_model_name)
query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)

cross_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
cross_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")

test_data_path = 'data/mimic3/handled/test_0105.json'
list_test_samples = []
with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            list_test_samples.append(line)

list_test_samples = list_test_samples[50:] #gpt-4o 先50个看一下




#----------rag----------查询流程函数----------分割线----------
def generate_embeddings(texts, model, tokenizer, max_length=512):
    """Generate embeddings for a list of texts using a pre-trained model."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

def load_chunks_to_memory(file_path, true_index):
    """Preload all chunks from a file into memory as a dictionary."""
    chunk_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
                record = json.loads(line)
                chunk_map[record["id"]] = record["content"]
    content = chunk_map[true_index]
    return content

def query_faiss_index(query_text, index_path, model, tokenizer, top_k=5):
    """Query a FAISS index and return the top-k results."""
    index = faiss.read_index(index_path)
    with open(index_path + ".map", "r", encoding="utf-8") as f:
        id_to_file_map = json.load(f)
    query_embedding = generate_embeddings([query_text], model, tokenizer)
    _, indices = index.search(query_embedding, top_k) #分别是距离和index
    print("indices:",indices)
    results = []
    for idx in indices[0]:
        true_index = list(id_to_file_map.keys())[idx]
        file_name = id_to_file_map[true_index]
        chunk_content = load_chunks_to_memory(file_name, true_index)
        results.append({"id": true_index, "content": chunk_content})
    return results

def search_docs_for_rag(query, index_path, query_model, query_tokenizer, cross_model, cross_tokenizer, top_k):
    #先粗略检索相近文档，再用MedCPT重排
    results = query_faiss_index(query, index_path, query_model, query_tokenizer, top_k)
    # for result in results:
    #     print(result)
    query_doc_pairs = [
        [query, doc['content']] for doc in results
    ]
    scores = cross_encode(query_doc_pairs, cross_model, cross_tokenizer)
    # 按降序排序并获取排序索引
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    final_searched_docs = []
    for idx in sorted_indices:
        # print(f"Document: {query_doc_pairs[idx][1]}; Score: {scores[idx]}")
        final_searched_docs.append(query_doc_pairs[idx][1])
    return final_searched_docs

# 生成交叉编码器评分(重排)
def cross_encode(pairs, model, tokenizer):
    inputs = tokenizer(
        pairs,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(dim=1)
    return logits

def find_conflict_info(advices, top_k = 10): #20250112添加药物冲突知识库
    index_path = "/Code-for-DDI/RAGents4DDI/faiss/faiss_index_ddi_info.idx"
    prompt_text = f"Find conflicts of the medicine advices <{advices}>"
    searched_doc = search_docs_for_rag(prompt_text, index_path, query_model, query_tokenizer, cross_model, cross_tokenizer, top_k)
    searched_docs = "Drug-drug interaction side effects: "
    for i in range(top_k):
        searched_docs = searched_docs + "/n" + searched_doc[i]
    # searched_docs = searched_doc[0] + "/n" + searched_doc[1] + "/n" + searched_doc[2] + "/n" + searched_doc[3] + "/n" + searched_doc[4]
    return searched_docs
#-------------------------------------------------------

#------------业务函数------------分割线-------------------
def atc_to_name(atc_code, atc_code_to_name):
   
    """
    将 ATC 编码转换为 ATC 名称。
    如果 ATC 编码不在字典中，返回 NaN。
    """
    return atc_code_to_name.get(atc_code, 'NaN')
def get_restrict_instruct():
    voc_dir = "/llm/Code-for-DDI/LEADER-pytorch-master/data/mimic4/handled/voc_final.pkl" #处理数据集的时候构建好的
    ehr_tokenizer = EHRTokenizer(voc_dir)
    who_df = pd.read_csv('/llm/Code-for-DDI/auxiliary/WHO ATC-DDD 2024-07-31.csv')  
    atc_code_to_name = dict(zip(who_df['atc_code'], who_df['atc_name']))
    # 创建一个新字典，存储 ATC 编码到名称的映射
    atc_code_to_name_result = {}
    # 遍历 atc_code_dict 中的每一项
    for atc_code in ehr_tokenizer.med_voc.word2idx:
        # 调用 atc_to_name 函数获取 ATC 名称
        atc_name = atc_to_name(atc_code, atc_code_to_name)
        atc_code_to_name_result[atc_code] = atc_name
    table_header = "| drug_atc_code | drug_atc_name                                      |\n|----------|-----------------------------------------------|"
    table_rows = "\n".join(f"| {code}     | {name} |" for code, name in atc_code_to_name_result.items())
    formatted_drug_table = f"{table_header}\n{table_rows}"
    RESTRICT_INSTRUCT= f"When you give medication recommendations, please choose the recommended medication from the 124 medications below : <{formatted_drug_table}>"
    return RESTRICT_INSTRUCT

def getlist(content): #转换为列表
    if type(content) == str:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            transformed_content = match.group(0)
            content = ast.literal_eval(transformed_content)
        else:
            print("未找到匹配的列表")
    return content
   
def get_conflict_dict(assign_list):
    result = {}
    for item in assign_list:
        expert, numbers_str = item.split(":")
        expert = expert.strip()
        numbers_str = numbers_str.strip()
        if len(numbers_str) > 0:
            parts = numbers_str.split(",")
            numbers = []
            for part in parts:
                part = part.strip()  # 去除空格
                if part.isdigit():   # 检查是否为纯数字字符串
                    numbers.append(int(part))
                else:
                    part = part[-1]
                    print(part)
                    numbers.append(int(part))
            result[expert] = numbers
    return result

def find_conflict_info(advices, top_k = 10): #20250112添加药物冲突知识库
    index_path = "/llm/Code-for-DDI/RAGents4DDI/faiss/faiss_index_ddi_info.idx"
    prompt_text = f"Find conflicts of the medicine advices <{advices}>"
    searched_doc = search_docs_for_rag(prompt_text, index_path, query_model, query_tokenizer, cross_model, cross_tokenizer, top_k)
    searched_docs = "Drug-drug interaction side effects: "
    for i in range(top_k):
        searched_docs = searched_docs + "/n" + searched_doc[i]
    return searched_docs

def generate_advices_from_single(single_chat_list, agent_specialists, conflicts, first_medication):
    print("生成单个意见")
    advices_from_single = "Recommendations addressing the drug conflicts:"
    for conflict_idx, specialist_item in single_chat_list.items():
        for agent_specialist in agent_specialists:
            if agent_specialist.role == specialist_item[0]:
                conflict_info_rag = find_conflict_info(conflicts[conflict_idx-1])
                prompt = f'''Following is the medication plan for a patient: <{first_medication}>. 
                    And here are some infomation you can refer to: <{conflict_info_rag}>. 
                    And here is one of the identified conflicts: <{conflicts[conflict_idx-1]}>
                    Based on this, review the conflict relevant to your specialty. 
                    Evaluate the current medication plan and provide a recommendation addressing the identified conflict.
                    Clearly explain the reasoning behind your recommendation and any adjustments to the medication plan.
                    
                    Note: Only output the adjustment with reason, such as add or delete drug.
                    Output format:  
                    1. **Delete drugs and reasons:** 
                    2. **Add drugs and reasons:**  

                '''
                advice_from_single = f"Advive from {agent_specialist.role} regarding conflict {conflict_idx}: \n" + agent_specialist.chat(prompt)
                print(advice_from_single)
                advices_from_single += "\n" + advice_from_single + "\n---"
    return advices_from_single

#生成多个的--聊天室
def judge_consesus(agent_mediator, advices_from_multi, agent_mediator_mes_text):
    agent_mediator.messages = agent_mediator.messages[:2] #####
    consesus_comparison = "\n---\n".join(
        [f"{item}" for index, item in enumerate(advices_from_multi)]
    )
    prompt = f'''
    In this round of discussions, clinical experts proposed advices as follows: <{consesus_comparison}>.
    The definition of consensus is that several clinical experts have exactly the same opinion.
    Your job is to determine if they have reached a consensus.
    Only respond with 'yes' or 'no'.

    Example 1:
    Specialist 1: Delete drug A. Add drug B.
    Specialist 2: Delete drug A. Add drug C.
    Consensus output: 'no'

    Example 2:
    Specialist 1: Delete drug A. Add drug B.
    Specialist 2: Delete drug A. Add drug B.
    Consensus output: 'yes'

    '''
    consensus_reached = agent_mediator.chat(prompt)
    agent_mediator_mes_text += agent_mediator.messages[2:]
    return consensus_reached, agent_mediator_mes_text

def generate_first_round_advices(prompt, multi_chat_members):    
    initial_advices = [] #聊天室的成员依次提出的建议
    print(multi_chat_members)
    for multi_chat_member in multi_chat_members:
        multi_chat_member.messages = multi_chat_member.messages[:2] #####
        advice = multi_chat_member.chat(prompt)
        print(f"Advice from {multi_chat_member.role}:", advice)
        advice_from_multi = f"{multi_chat_member.role} proposed:" + advice
        initial_advices.append(advice_from_multi)
    return initial_advices

def generate_next_round_advices(multi_chat_members, advices_from_multi):    
    next_round_advices = [] 
    other_advices = []
    for i in range(len(advices_from_multi)):
        # 排除当前索引的元素，组合其他元素
        other_advice = "".join([advices_from_multi[j] if j != i else "" for j in range(len(advices_from_multi))])
        other_advices.append(other_advice)
    for index, multi_chat_member in enumerate(multi_chat_members):
        advice = multi_chat_member.chat(other_advices[index] + '''
        Do you think there are any recommendations among these that are more reasonable than the ones you proposed? 
        You can also put forward new modification recommendation.
        ''')
        print(f"Advice from {multi_chat_member.role}:", advice)
        advice_from_multi = f"{multi_chat_member.role} proposed:" + advice
        next_round_advices.append(advice_from_multi)
    return next_round_advices

def consensus_process(agent_mediator, multi_chat_members, initial_advices):
    advices_from_multi = initial_advices  # 初始化建议列表
    max_rounds = 5  # 最大循环轮次，避免无限循环
    current_round = 0
    agent_mediator_mes_text = []
    while current_round < max_rounds:
        consensus_reached, agent_mediator_mes_text = judge_consesus(agent_mediator, advices_from_multi, agent_mediator_mes_text)
        if "yes" in consensus_reached.lower():
            consensus_prompt = "What is the final consensus?"
            consensus_advice = agent_mediator.chat(consensus_prompt)
            print("Final consensus reached:", consensus_advice)
            return consensus_advice, current_round, agent_mediator_mes_text 
        elif "no" in consensus_reached.lower():
            current_round += 1
            print(f"Round {current_round}: No consensus reached. Generating next round advices...")
            # print("agent_mediator messages:", agent_mediator.messages)
            advices_from_multi = generate_next_round_advices(multi_chat_members, advices_from_multi)
        else:
            print("Unexpected response from judge_consesus:", consensus_reached)
            return "No consensus"  
    if current_round >= max_rounds:
        print("Max rounds reached. No consensus achieved.")
        consensus_prompt = "What is the most possible final consensus?"
        consensus_advice = agent_mediator.chat(consensus_prompt)
        print("Final possible consensus:", consensus_advice)
        return consensus_advice, current_round, agent_mediator_mes_text
    
def generate_consensus_advice(conflict_idx, specialists_item, agent_specialists, conflicts, first_medication):
    multi_chat_members = [] #找到该聊天室的成员，再遍历生成修改意见
    for specialist_item in specialists_item: #['Nephrologist', 'Rheumatologist']
        for agent_specialist in agent_specialists:
            if agent_specialist.role == specialist_item:
                agent_specialist.messages[:2] #为了减少上下文
                multi_chat_members.append(agent_specialist)
    conflict_info_rag = find_conflict_info(conflicts[conflict_idx-1])
    prompt = f'''
                    Following is the medication plan for a patient: <{first_medication}>. 
                    And here are some infomation you can refer to: <{conflict_info_rag}>. 
                    And here is one of the identified conflicts: <{conflicts[conflict_idx-1]}>
                    Based on this, review the conflict relevant to your specialty. 
                    Evaluate the current medication plan and provide a recommendation addressing the identified conflict.
                    Clearly explain the reasoning behind your recommendation and any adjustments to the medication plan.
                    
                    Note: Only output the adjustment with reason.
                    Output format:  
                    1. **Adjustments:**  

                '''
    initial_advices = generate_first_round_advices(prompt, multi_chat_members) #聊天室的成员依次提出的建议 
    #聊天室调停者--判断大家是否达成了共识
    MEDIATOR_INSTRUCT = "You are a professional mediator in a chat room where several clinical experts are proposing their suggestions on drug interactions. Your job is to determine if they have reached a consensus, and provide final discussion result to the medication plan adjustment."
    agent_mediator = Agent_gpt(MEDIATOR_INSTRUCT, 'mediator') 
    consensus_advice, current_round, agent_mediator_mes_text = consensus_process(agent_mediator, multi_chat_members, initial_advices)
    return consensus_advice, current_round, agent_mediator_mes_text



def generate_advices_from_multi(multi_chat_list, agent_specialists, conflicts, first_medication):
    print("拉聊天室讨论")
    advices_from_multi = ""
    advices_from_multi_rounds = []
    advices_from_multi_text = []
    for conflict_idx, specialists_item in multi_chat_list.items():
        print(type(conflict_idx))
        print(f"冲突{conflict_idx}聊天室")
        consensus_advice, current_round, agent_mediator_mes_text = generate_consensus_advice(conflict_idx, specialists_item, agent_specialists, conflicts, first_medication)
        advices_from_multi += f"Recommendations for conflict {conflict_idx}: " + consensus_advice + "\n---"
        advices_from_multi_rounds.append(current_round)
        advices_from_multi_text.append(agent_mediator_mes_text)
    return advices_from_multi, advices_from_multi_rounds, advices_from_multi_text

def initialize_agents():
    print("智能体初始化")
    #全科医生
    TASK_PLANNER_INSTRUCT = "You are an experienced general practitioner, your job is to assess the patient's condition and make decisions."
    agent_task_planner = Agent_gpt(TASK_PLANNER_INSTRUCT, 'practitioner') 

    #数据整理师
    DATA_COMPILER_INSTRUCT = "You are a helpful data organizer, your job is to convert the doctor's spoken content into structured data."
    agent_data_compiler = Agent_gpt(DATA_COMPILER_INSTRUCT, 'compiler')

    #药师
    PHARMACIST_INSTRUCT = "You are a experienced pharmacist, your job is to review the prescribed medications, assess for drug interactions, and recommend the best pharmacological approach."
    agent_pharmacist = Agent_gpt(PHARMACIST_INSTRUCT, 'pharmacist')
    return agent_task_planner, agent_data_compiler, agent_pharmacist

def create_agent_specialists(organized_MDT_plan, RESTRICT_INSTRUCT):
    if type(organized_MDT_plan) == str:
        specialists  = ast.literal_eval(organized_MDT_plan)
    else:
        specialists  = organized_MDT_plan
    if(len(specialists) < 2): 
        specialists = ["practitioner"]
    print(specialists)
    # 创建专科医生们
    SPECIALIST_INSTRUCT = "You are a <SUBJECT> specialist, your job is to provide expertise in your specific field to contribute to the treatment plan based on the patient's condition."
    # 用字典存储每个 specialist 的代理对象
    agent_specialists = []
    for index, specialist in enumerate(specialists):
        # 替换占位符
        specialist_instruct_new = SPECIALIST_INSTRUCT.replace("<SUBJECT>", specialist) + RESTRICT_INSTRUCT
        # 创建代理对象并存储到字典
        agent_name = f"{specialist}"  # 动态生成代理名称
        agent_specialist = Agent_gpt(specialist_instruct_new, agent_name)
        agent_specialists.append(agent_specialist)
        print(f"Created agent: {agent_name} for {specialist}")
    return agent_specialists

def recruit_MDT(patient_condition, agent_data_compiler, agent_task_planner, RESTRICT_INSTRUCT):
    print("首次招募MDT")
    #初始输入，确定是否需要多学科，及可能需要的领域专家 
    initial_input = "Following is a patient's condition: {<CONDITION>}. Please determine if multidisciplinary medication is required, and identify the relevant specialist disciplines."
    initial_input = initial_input.replace("<CONDITION>", patient_condition)
    MDT_plan = agent_task_planner.chat(initial_input)
    # print('MDT_plan:', MDT_plan)
    #多学科整理成专家list
    organized_MDT_plan_prompt = "Following is the treatment strategy: {<STRATEGY>}. Please Extract and organize the specialist disciplines mentioned in the treatment strategy into a clean list format, like this ['Pulmonologist', 'Neurologist', ...]. Note: Only return the list."
    organized_MDT_plan_prompt = organized_MDT_plan_prompt.replace("<STRATEGY>", MDT_plan)
    organized_MDT_plan = agent_data_compiler.chat(organized_MDT_plan_prompt)
    # print(organized_MDT_plan) 
    #动态构建多智能体（第一次）
    agent_specialists = create_agent_specialists(organized_MDT_plan, RESTRICT_INSTRUCT) 
    print(len(agent_specialists))
    return agent_specialists, MDT_plan

# 遍历所有专科医生智能体，获取用药建议(仅返回药品code)
def generate_medicine_advise(agent_specialists, practitioner, agent_data_compiler, med_advise_prompt, organize_med_advise_prompt, atc_code_to_name):
    specialist_advices = []
    specialist_advices_text = []
    for index, agent_specialist in enumerate(agent_specialists):
        med_advise = agent_specialist.chat(med_advise_prompt)
        med_advise_text = f"Unorganized Advice from Specialist {agent_specialist.role}: {med_advise}"
        specialist_advices_text.append(med_advise_text)
        print(f"Unorganized Advice from Specialist {agent_specialist.role}: {med_advise}\n")  # 打印每个专科医生的用药建议
        organize_med_advise_prompt_to_use = organize_med_advise_prompt.replace("<med_advise>", med_advise)
        agent_data_compiler.messages = agent_data_compiler.messages[:4] #仅留一条system的和一对对话，防止token不够用
        med_advise = agent_data_compiler.chat(organize_med_advise_prompt_to_use)
        # med_advise有几种情况，先转字符串，再用正则匹配整理成list
        med_advise = str(med_advise)
        med_advise = re.findall(r'[A-Z]\d{2}[A-Z]', med_advise)
        specialist_advices += med_advise
    first_medication_codes = list(set(specialist_advices)) #第一份综合用药
    med_advise_code_and_name = {}
    for med_code in first_medication_codes:
        med_name = atc_to_name(med_code, atc_code_to_name)
        med_advise_code_and_name[med_code] = med_name
    first_medication = "\n".join(f"{name}" for code, name in med_advise_code_and_name.items())
    print("first_medication_codes:", first_medication_codes)
    print("first_medication:", first_medication)
    return first_medication_codes, first_medication, specialist_advices_text


def conflict_revise(specialist_advices, agent_pharmacist, type="test"): 
    if type == "analyze":
        agent_pharmacist.messages = agent_pharmacist.messages[:4]
    
    revise_prompt = f'''
        Here are advices on conflicts from several specialists: <{specialist_advices}>.
       
        Your task is to:
        Basing on advices proposed by specialists and previous medication recommendation, organize a revised medication recommendation.

        Only output the revised medication recommendation.  

        Output format:
        1. **Revised Medication Recommendation:**
        2. **Additional Comments:**  
    '''
    revised_result = agent_pharmacist.chat(revise_prompt)

    if type == "analyze":
        agent_pharmacist.messages = agent_pharmacist.messages[:4]
    return revised_result


def vote(consolidated_new_medication_plan, agent_specialists, vote_times):
    vote_times += 1
    new_med_plan_agree_or_not_prompt = f'''Following is the revised medication plan and reason regarding previous conflicts: <{consolidated_new_medication_plan}>.
        Do you agree with this revised medication plan?
        Please respond with 'Yes' or 'No', and provide any recommendation if necessary.
        **Output Format:**  
        1. **Short Response:**  
        'Yes' or 'No' 

        2. **Additional Comments:**   
    '''
    agree_or_not_list = [] 
    for index, agent_specialist in enumerate(agent_specialists):
        agree_or_not = agent_specialist.chat(new_med_plan_agree_or_not_prompt)
        short_res_match = re.search(r"Short Response(.+?)Additional Comments", agree_or_not, re.DOTALL)
        short_res = short_res_match.group(0).strip() if short_res_match else ""
        addi_comment_match = re.search(r"Additional Comments.*", agree_or_not, re.DOTALL)
        addi_comment = addi_comment_match.group(0).strip() if addi_comment_match else ""
        agree_or_not_list.append({'agent': agent_specialist.role, 'short_res': short_res, 'comment': addi_comment})
    print(agree_or_not_list)
    yes_count = 0
    no_count = 0
    for index, agree_or_not in enumerate(agree_or_not_list):
        text = agree_or_not['short_res']
        if (count_no_occurrences(text) >= 1):
            no_count += 1
            agree_or_not_list[index]['short_res'] = 'No'
        elif (count_yes_occurrences(text) >= 1):
            yes_count +=1
            agree_or_not_list[index]['short_res'] = 'Yes'
    return yes_count, no_count, agree_or_not_list, vote_times

def count_no_occurrences(text):
        return text.lower().count("no")
def count_yes_occurrences(text):
        return text.lower().count("yes")

def solve_disagreement(yes_count, no_count, agree_or_not_list, agent_specialists, agent_task_planner, agent_pharmacist, consolidated_new_medication_plan, vote_times):
    while no_count > 0: #循环 
        print('药师根据no的意见重新生成新药单')
        comments_for_consideration = [{'agent': item['agent'], 'comment': item['comment']} for item in agree_or_not_list if item['short_res'] == 'No']
        
        advice_comparison = "\n---\n".join(
            [f"{item['agent']} proposed: {item['comment']}" for index, item in enumerate(comments_for_consideration)]
        )
        disagree_prompt = f'''There are {no_count} specialists disagree with your revised medication plan.
        Here are their comments: <{advice_comparison}>.
        Please organize a new revised medication plan with consideration on the comments.

        Output format:
        1. **Revised Medication Recommendation:**
        2. **Additional Comments:**  
        '''
        agent_pharmacist.messages = [agent_pharmacist.messages[0]] + [agent_pharmacist.messages[1]] + agent_pharmacist.messages[-2:]
        new_revised_plan = agent_pharmacist.chat(disagree_prompt)
        consolidated_new_medication_plan = new_revised_plan
        yes_count, no_count, agree_or_not_list, vote_times = vote(new_revised_plan, agent_specialists, vote_times)
   
    return consolidated_new_medication_plan, vote_times


def get_drug_list(final_res):
    final_compile_prompt = f'''You are tasked with processing a given medication plan <{final_res}>. 
    Please organize the medication plan into a final use version.
    Output in clean list format, like this ['drug 1', 'drug2', ...].
    Note: Only return the list.
    '''
    DATA_COMPILER_INSTRUCT = "You are a helpful data organizer, your job is to convert the doctor's spoken content into structured data."
    agent_data_compiler = Agent_gpt(DATA_COMPILER_INSTRUCT, 'compiler')
    rrres = agent_data_compiler.chat(final_compile_prompt) #药方中药物名称list
    rrres = str(rrres)
    # 使用正则表达式提取列表,以防输出中有其他文字
    if type(rrres) == str:
        print(rrres)
        match = re.search(r"\[.*\]", rrres, re.DOTALL)  # 匹配以 '[' 开头，以 ']' 结尾的内容，支持跨行匹配
        if match:
            rrres = match.group(0)  # 获取匹配到的内容
        else:
            #加补丁
            def safe_eval(input_str):
                try:
                    return ast.literal_eval(input_str)
                except (ValueError, SyntaxError) as e:
                    DATA_COMPILER_INSTRUCT = "You are a helpful data organizer, your job is to convert the doctor's spoken content into structured data."
                    agent_data_compiler = Agent_gpt(DATA_COMPILER_INSTRUCT, 'compiler')
                    agent_data_compiler.chat(DATA_COMPILER_INSTRUCT)
                    prompt = f"Please transform this string <{input_str}> into a standard list like this ['C07AB', 'C09B', ...]. Note: Only return the list."
                    transformed_input = agent_data_compiler.chat(prompt)
                    if type(transformed_input) == str:
                        #再打个补丁
                        match = re.search(r"\[.*\]", transformed_input, re.DOTALL)  # 匹配以 '[' 开头，以 ']' 结尾的内容，支持跨行匹配
                        if match:
                            transformed_input = match.group(0)  # 获取匹配到的内容
                            print("提取的列表:", transformed_input)
                        else:
                            print("未找到匹配的列表", transformed_input)
                        transformed_input = ast.literal_eval(transformed_input)
                    return transformed_input
            rrres = safe_eval(rrres)
            print("未找到匹配的列表")
    return rrres     


def transform_unknown_drugs(unknown_drugs):
    print(unknown_drugs, type(unknown_drugs))
     #转换为找到atc3的药物
    index_path = "/llm/Code-for-DDI/RAGents4DDI/faiss/faiss_index_drug2atc.idx"
    searched_docs = []
    for index, unknown_drug in enumerate(unknown_drugs):
        query_drug = "What's the most possible ATC drug names of the drug <<drug>> ?"
        query_drug = query_drug.replace("<drug>", unknown_drugs[index])
        searched_doc = search_docs_for_rag(query_drug, index_path, query_model, query_tokenizer, cross_model, cross_tokenizer, top_k=5)
        searched_docs.append(searched_doc[0])
        searched_docs.append(searched_doc[1])
    query_drug = f"What are the most possible specific ATC drug names of the drugs <{unknown_drugs}> ?"
    drug2atc_rag_prompt = f"Refer to the following information <{searched_docs}>. And answer {query_drug}. Output in a list format like this ['C08A', 'A09C', ...]. Note: Only return the list."
    
    PHARMACIST_INSTRUCT = "You are a experienced pharmacist, your job is to review the prescribed medications, assess for drug interactions, and recommend the best pharmacological approach."
    agent_pharmacist = Agent_gpt(PHARMACIST_INSTRUCT, 'pharmacist')
    agent_pharmacist.chat(PHARMACIST_INSTRUCT)

    transformed_unknown_drugs = agent_pharmacist.chat(drug2atc_rag_prompt)
    print('transformed_unknown_drugs',transformed_unknown_drugs, type(transformed_unknown_drugs))
    if type(transformed_unknown_drugs) == str:
        #补丁：正对字符串中有的编码只有单边引号的情况
        def safe_eval(input_str):
            try:
                return ast.literal_eval(input_str)
            except (ValueError, SyntaxError) as e:
                DATA_COMPILER_INSTRUCT = "You are a helpful data organizer, your job is to convert the doctor's spoken content into structured data."
                agent_data_compiler = Agent_gpt(DATA_COMPILER_INSTRUCT, 'compiler')
                agent_data_compiler.chat(DATA_COMPILER_INSTRUCT)
                prompt = f"Please transform this string <{input_str}> into a standard list like this ['C07AB', 'C09B', ...]. Note: Only return the list."
                transformed_input = agent_data_compiler.chat(prompt)
                if type(transformed_input) == str:
                    #再打个补丁
                    match = re.search(r"\[.*\]", transformed_input, re.DOTALL)  # 匹配以 '[' 开头，以 ']' 结尾的内容，支持跨行匹配
                    if match:
                        transformed_input = match.group(0)  # 获取匹配到的内容
                        print("提取的列表:", transformed_input)
                    else:
                        print("未找到匹配的列表", transformed_input)
                    transformed_input = ast.literal_eval(transformed_input)
                # print(transformed_input, "transformed_input")
                return transformed_input
        transformed_unknown_drugs  = safe_eval(transformed_unknown_drugs)
    return transformed_unknown_drugs
     
def process_drug2atc(drug_list):
    atc_file_path = "/llm/Code-for-DDI/auxiliary/WHO ATC-DDD 2024-07-31.csv"
    atc_df = pd.read_csv(atc_file_path)
    # 创建药物名称到 ATC 编码的映射字典
    atc2drug_dict = dict(zip(atc_df['atc_name'].str.lower(), atc_df['atc_code']))
    if type(drug_list) == str:
            drug_list  = ast.literal_eval(drug_list)
    atc_codes = [atc2drug_dict.get(drug.lower(), 'Unknown') for drug in drug_list]
    print("1:", atc_codes)
    known_drugs = [drug for drug, atc_code in zip(drug_list, atc_codes) if atc_code != 'Unknown']
    unknown_drugs = [drug for drug, atc_code in zip(drug_list, atc_codes) if atc_code == 'Unknown']
    print("2:", unknown_drugs)
    # 把atc2drug里找不到的药物用RAG，找出最符合、接近的药物
    if unknown_drugs == []:
        transformed_unknown_drugs = []
    else:
        transformed_unknown_drugs = transform_unknown_drugs(unknown_drugs)
    print("3:", transformed_unknown_drugs)
    filtered_list = [item for item in atc_codes if item != "Unknown"]
    atc_codes = filtered_list + transformed_unknown_drugs
    return atc_codes

def get_restrict_priority_instruct(top_n): #124种药物中挑选+优先级限制
    voc_dir = "/llm/Code-for-DDI/LEADER-pytorch-master/data/mimic4/handled/voc_final.pkl" #处理数据集的时候构建好的
    ehr_tokenizer = EHRTokenizer(voc_dir)
    who_df = pd.read_csv('/llm/Code-for-DDI/auxiliary/WHO ATC-DDD 2024-07-31.csv')  
    atc_code_to_name = dict(zip(who_df['atc_code'], who_df['atc_name']))
    # 创建一个新字典，存储 ATC 编码到名称的映射
    atc_code_to_name_result = {}
    # 遍历 atc_code_dict 中的每一项
    # 进行排序
    sorted_keys = sorted(ehr_tokenizer.med_voc.idx_num.keys(), key=lambda x: ehr_tokenizer.med_voc.idx_num[x], reverse=True)
    new_word2idx = {key: ehr_tokenizer.med_voc.idx_num[key] for key in sorted_keys}
    for atc_code in new_word2idx:
        atc_name = atc_to_name(atc_code, atc_code_to_name)
        atc_code_to_name_result[atc_code] = atc_name
    # 添加优先级编号
    table_header = "| Rank | drug_atc_code | drug_atc_name                                      |\n|----------|---------------|----------------------------------------------------|"
    table_rows = "\n".join(
        f"| {i+1}        | {code}         | {name} |" 
        for i, (code, name) in enumerate(atc_code_to_name_result.items())
    )
    formatted_drug_table = f"{table_header}\n{table_rows}"    
    RESTRICT_INSTRUCT= f'''Following is a medication table containing 124 medications ranked by recommendation frequency: <{formatted_drug_table}>.
    When you give medication recommendations, please choose the medication from the medication table.
    When you are facing conflicts between two drugs, take consideration on following steps:
    <1. First, if the rank of the drugs are both in top {top_n}, do not make any adjustments.
    2. Consider which drug's rank is lower, then, when you make adjustments, consider detele or replace the lower rank drug. >
    '''
    return RESTRICT_INSTRUCT
#-------------------------------------------------------
    





#------------关键流程函数------------分割线-------------------
def get_medicine_recommendation(vote_times, list_test_sample, top_n):
    # RESTRICT_INSTRUCT = get_restrict_instruct()
    RESTRICT_INSTRUCT = get_restrict_priority_instruct(top_n) 
    who_df = pd.read_csv('/llm/Code-for-DDI/auxiliary/WHO ATC-DDD 2024-07-31.csv')  
    atc_code_to_name = dict(zip(who_df['atc_code'], who_df['atc_name']))
    #初始化
    agent_task_planner, agent_data_compiler, agent_pharmacist = initialize_agents()
    patient_condition = list_test_sample['input'] 
    #招募MDT
    agent_specialists, MDT_plan = recruit_MDT(patient_condition, agent_data_compiler, agent_task_planner, RESTRICT_INSTRUCT)
    #转换一下格式，写入结果
    agent_specialists_str_list = []
    for agent_specialist in agent_specialists:
        role = agent_specialist.role
        agent_specialists_str_list.append(role)
    med_advise_prompt = patient_condition
    #整理用药建议->仅返回药品ATC3编码--不同于之前
    organize_med_advise_prompt = "Following is a medication recommendation given by a specialist: <med_advise>. Please extract the drug code from the medication recommendations provided for the patient's treatment plan. Only output the code of the medications without additional information."
    # 各专科医生生成用药建议 -> 返回药品名称
    first_medication_codes, first_medication, specialist_advices_text = generate_medicine_advise(agent_specialists, agent_task_planner, agent_data_compiler, med_advise_prompt, organize_med_advise_prompt, atc_code_to_name)
    #检测冲突-- llama3.18b 会输出很多重复的冲突，换个基座试一下
    conflict_info_rag = find_conflict_info(first_medication)
    # medcorpus_rag_docs = generate_medcorpus_rag(first_medication, "identify")
    conflict_identify_prompt = f"""
            Read the documents <{conflict_info_rag}>.
            The following is a medication recommendation for the patient's condition: <{patient_condition}>.  
        
            Your role as a clinical pharmacist is to:

            1. **Analyze** the provided medication recommendations.  
            2. Identify any **conflicting medications** or **potential contraindications** among the recommended drugs.  
            3. Sort the conflicts by severity of side effects

            For all identified conflicts, clearly explain:  
            - The **conflicting drugs** and the nature of the conflict (e.g., drug-drug interaction, contraindicated condition).   

            Medication recommendation:  
            {first_medication}

            Output format:  
            **Potential Conflicts or Contraindications:**  
            - Conflict 1: [Conflict Description 1]  
            - Conflict 2: [Conflict Description 2]   
        """
    conflict_check_result = agent_pharmacist.chat(conflict_identify_prompt)
    #招募讨论组，分配论点
    further_discussion_prompt = f"""
            Medication conflicts identified by Pharmacist: <{conflict_check_result}>.
            As the general physician, your task is to review the pharmacist's identified conflicts regarding medication recommendation for the patient's condition: <{patient_condition}>.  

            **Your responsibilities:**  
            1. Carefully review the conflicts.

            2. Determine the **specific specialists** who need to be involved for further consultation.  

            4. Provide a clear summary that includes:  
            - **Conflicts:** The conflicts requiring further discussion.  
            - **Specialists to Consult:** A list of relevant specialists needed to resolve these issues (e.g., Cardiologist, Pulmonologist, Nephrologist).  
            - **Assign conflicts:** Assign conflicts that are related to the specialist.
            

            **Output Format:**  
            1. **Conflicts:**  
            - Conflict 1: [Description of the conflict]  
            - Conflict 2: [Description of the conflict]  

            2. **Specialists to Consult:**  
            - Specialist 1: [Reason for consultation]  
            - Specialist 2: [Reason for consultation]  

            3. **Assign Conflicts:** 
            - Specialists 1: [Conflict numbers related to Specialist 1]
            - Specialists 2: [Conflict numbers related to Specialist 2]

        """
    further_discussion = agent_task_planner.chat(further_discussion_prompt)
    print(further_discussion) 
    #提取 冲突、讨论组、分配

    # print(further_discussion) 
    conflicts = re.search(r"Conflicts(.*?)Specialists to Consult", further_discussion, re.DOTALL).group(1)
    # print(len(conflicts), conflicts)
    pattern = r'(Conflict \d+)(.*?)(?=Conflict \d+|$)'
    matches = re.findall(pattern, conflicts, re.DOTALL)
    conflicts = []
    for match in matches:
        conflict_part = match[0]  # Conflict n
        content = match[1].strip()  # Conflict 之间的内容
        conflicts.append(conflict_part + content)  
    print(len(conflicts),conflicts)
    assign_conflicts_match = re.search(r"Assign Conflicts.*", further_discussion, re.DOTALL)
    assign_conflicts = assign_conflicts_match.group(0).strip() if assign_conflicts_match else ""
    # print(assign_conflicts)
    assign_conflict_prompt = f'''Now that several doctors are being assigned conflict-specific discussion tasks, here are the assignment results <{assign_conflicts}>. 
    Please organize the distribution results into list format like this ["Specialist: conflict numbers", "Specialist: conflict numbers"].

    Only output the list.

    '''
    result = agent_data_compiler.chat(assign_conflict_prompt)
    assign_list = getlist(result)
    conflicts_dict = get_conflict_dict(assign_list)
    # print(conflicts_dict)
    specialists_list = list(conflicts_dict.keys())
    #招募新团队
    agent_specialists = create_agent_specialists(specialists_list, RESTRICT_INSTRUCT)
    # print(specialists_list)
    conflict_to_specialists = {}
    for specialist, conflict_item in conflicts_dict.items():
        for conflict in conflict_item:
            # 如果冲突编号已经存在于新字典中，则添加专家名称
            if conflict in conflict_to_specialists:
                conflict_to_specialists[conflict].append(specialist)
            else:
                # 否则，创建一个新的列表
                conflict_to_specialists[conflict] = [specialist]

    single_chat_list = {}
    multi_chat_list = {}
    for conflict_idx, specialist_item in conflict_to_specialists.items():
        if len(specialist_item) == 1:
            single_chat_list[conflict_idx] = specialist_item
        else:
            multi_chat_list[conflict_idx] = specialist_item
    
    advices_from_single = generate_advices_from_single(single_chat_list, agent_specialists, conflicts, first_medication)
    print("0215----1:", multi_chat_list)
    advices_from_multi, advices_from_multi_rounds, advices_from_multi_text = generate_advices_from_multi(multi_chat_list, agent_specialists, conflicts, first_medication)

    specialist_advices = advices_from_single + advices_from_multi
    # print(specialist_advices)
    #------#-------#--------node2的对照组--------------------------
    #1.只有一个全科医生---固定单智能体进行决策，不考虑动态生成和分配；
    # single_fixed_specialist = create_agent_specialists(["General practitioner"], RESTRICT_INSTRUCT)
    # fixed_single_chat_list = {}
    # for index, item in enumerate(conflicts):
    #     fixed_single_chat_list[index+1] = ["General practitioner"]
    # fixed_advices_from_single = generate_advices_from_single(fixed_single_chat_list, single_fixed_specialist, conflicts, first_medication)
    # #2. 两人讨论（全科医生、药师） ---固定多智能体进行决策，不考虑动态生成和分配；
    # multi_fixed_specialists = create_agent_specialists(["General practitioner", "Pharmacist"], RESTRICT_INSTRUCT)
    # fixed_multi_chat_list = {}
    # for index, item in enumerate(conflicts):
    #     fixed_multi_chat_list[index+1] = ["General practitioner", "Pharmacist"]
    # print("0215:----2", fixed_multi_chat_list)
    # fixed_advices_from_multi, fixed_advices_from_multi_rounds, fixed_advices_from_multi_text = generate_advices_from_multi(fixed_multi_chat_list, multi_fixed_specialists, conflicts, first_medication)
    # fixed_conflict_revise_res_from_single = conflict_revise(fixed_advices_from_single, agent_pharmacist, type="analyze")
    # fixed_conflict_revise_res_from_multi = conflict_revise(fixed_advices_from_multi, agent_pharmacist, type="analyze")
    # -------------------------------------------------------------------

    #------#-------#--------亚群分析--------------------------
    # conflict_revise_res_from_single = conflict_revise(advices_from_single, agent_pharmacist, type="analyze")
    # conflict_revise_res_from_multi = conflict_revise(advices_from_multi, agent_pharmacist, type="analyze")
    # -------------------------------------------------------------------
    #药师根据团队意见生成新的用药建议
    conflict_revise_res = conflict_revise(specialist_advices, agent_pharmacist)
    # print("第二次用药意见生成:", conflict_revise_res)
    consolidated_new_medication_plan = conflict_revise_res
    #提取药师整理出的针对冲突的'新药方和原因'，问组内专家们是否同意(相当于投票)
    yes_count, no_count, agree_or_not_list, vote_times = vote(consolidated_new_medication_plan, agent_specialists, vote_times)
    final_res, vote_times = solve_disagreement(yes_count, no_count, agree_or_not_list, agent_specialists, agent_task_planner, agent_pharmacist, consolidated_new_medication_plan, vote_times)
    second_drug_list = get_drug_list(conflict_revise_res)
    second_drug_atc_codes = process_drug2atc(second_drug_list)
    final_drug_list = get_drug_list(final_res)
    final_drug_atc_codes = process_drug2atc(final_drug_list)
    #------#-------#--------亚群分析--------------------------  
    # analyze_single_drug_list = get_drug_list(conflict_revise_res_from_single)
    # analyze_single_drug_atc_codes = process_drug2atc(analyze_single_drug_list)
    # analyze_multi_drug_list = get_drug_list(conflict_revise_res_from_multi)
    # analyze_multi_drug_atc_codes = process_drug2atc(analyze_multi_drug_list)
    # -------------------------------------------------------------------
    #------#-------#--------node2的对照组--------------------------
    # #1. fix single
    # fixed_single_vote_times = 0
    # consolidated_new_medication_plan = fixed_conflict_revise_res_from_single
    # yes_count, no_count, agree_or_not_list, fixed_single_vote_times = vote(consolidated_new_medication_plan, single_fixed_specialist, fixed_single_vote_times)
    # fixed_single_final_res, fixed_single_vote_times = solve_disagreement(yes_count, no_count, agree_or_not_list, single_fixed_specialist, agent_task_planner, agent_pharmacist, consolidated_new_medication_plan, fixed_single_vote_times)
    # fixed_single_second_drug_list = get_drug_list(fixed_conflict_revise_res_from_single)
    # fixed_single_second_drug_atc_codes = process_drug2atc(fixed_single_second_drug_list)
    # fixed_single_final_drug_list = get_drug_list(fixed_single_final_res)
    # fixed_single_final_drug_atc_codes = process_drug2atc(fixed_single_final_drug_list)
    # #2. fix multi
    # fixed_multi_vote_times = 0
    # consolidated_new_medication_plan = fixed_conflict_revise_res_from_multi
    # yes_count, no_count, agree_or_not_list, fixed_multi_vote_times = vote(consolidated_new_medication_plan, multi_fixed_specialists, fixed_multi_vote_times)
    # fixed_multi_final_res, fixed_multi_vote_times = solve_disagreement(yes_count, no_count, agree_or_not_list, multi_fixed_specialists, agent_task_planner, agent_pharmacist, consolidated_new_medication_plan, fixed_multi_vote_times)
    # fixed_multi_second_drug_list = get_drug_list(fixed_conflict_revise_res_from_multi)
    # fixed_multi_second_drug_atc_codes = process_drug2atc(fixed_multi_second_drug_list)
    # fixed_multi_final_drug_list = get_drug_list(fixed_multi_final_res)
    # fixed_multi_final_drug_atc_codes = process_drug2atc(fixed_multi_final_drug_list)
    # -------------------------------------------------------------------

    final_output = {
    "patient_condition": patient_condition,
    "target_drug": {
        "target": list_test_sample["target"],
        "target_drug_code": list_test_sample["drug_code"]
    },
    "MDT_list": agent_specialists_str_list,
    "MDT_plan": MDT_plan, #存一下
    "first_medicine": { #初始药方
        "medicine_list": first_medication, #药物名称
        "atc3_code_list": first_medication_codes, 
        "conflicts": conflicts, #冲突by llm
        "specialist_advices_text": specialist_advices_text
    },
    "first_discussion_group_medicine": { #初次募的讨论组的相关数据
        "advices_from_single": advices_from_single, #添加中间分析
        "advices_from_multi": advices_from_multi,
        "single_chat_list": single_chat_list, #分组情况
        "multi_chat_list": multi_chat_list,
        "advices_from_multi_rounds": advices_from_multi_rounds, #讨论组的聊天室为了达成共识需要的轮数
        "advices_from_multi_text": advices_from_multi_text, #讨论组的聊天室的聊天记录
        "discussion_times": vote_times, #最后的投票次数
    },
    #------#-------#--------亚群分析-------------------------- 
    # "single_discussed_medicine": { 
    #     "medicine_list": analyze_single_drug_list, #药物名称
    #     "atc3_code_list": analyze_single_drug_atc_codes, #
    # },
    # "multi_discussed_medicine": { 
    #     "medicine_list": analyze_multi_drug_list, #药物名称
    #     "atc3_code_list": analyze_multi_drug_atc_codes, #
    # },
    # -------------------------------------------------------------------
    #------#-------#--------node2的对照组--------------------------
    # "fixed_single_inprogress_record": {
    #     "fixed_advices_from_single": fixed_advices_from_single,
    #     "discussion_times": fixed_single_vote_times
    # },
    # "fixed_multi_inprogress_record": {
    #     "fixed_advices_from_multi": fixed_advices_from_multi,
    #     "discussion_times": fixed_multi_vote_times,
    #     "advices_from_multi_rounds": fixed_advices_from_multi_rounds, #讨论组的聊天室为了达成共识需要的轮数
    #     "advices_from_multi_text": fixed_advices_from_multi_text,
    # },
    # "fixed_single_second_medicine": { 
    #     "medicine_list": fixed_single_second_drug_list, #药物名称
    #     "atc3_code_list": fixed_single_second_drug_atc_codes, #
    # },
    # "fixed_single_final_medicine": { 
    #     "medicine_list": fixed_single_final_drug_list, #药物名称
    #     "atc3_code_list": fixed_single_final_drug_atc_codes, #
    # },
    # "fixed_multi_second_medicine": { 
    #     "medicine_list": fixed_multi_second_drug_list, #药物名称
    #     "atc3_code_list": fixed_multi_second_drug_atc_codes, #
    # },
    # "fixed_multi_final_medicine": { 
    #     "medicine_list": fixed_multi_final_drug_list, #药物名称
    #     "atc3_code_list": fixed_multi_final_drug_atc_codes, #
    # },
    # -------------------------------------------------------------------

    "second_medicine": { 
        "medicine_list": second_drug_list, #药物名称
        "atc3_code_list": second_drug_atc_codes, #
        "text": conflict_revise_res
    },
    "final_medicine": { 
        "medicine_list": final_drug_list, #药物名称
        "atc3_code_list": final_drug_atc_codes, #
        "text": final_res
    }
}
    #将结果写入文件
    output_path = f"/llm/Code-for-DDI/RAGents4DDI/output/2025-03-19_gpt4o_50-_node2_rag_mimic3.json"
    # 新的数据
    new_data = final_output  
    # 读取现有数据并追加
    try:
        with open(output_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # 读取现有数据，文件必须为标准 JSON 格式
    except FileNotFoundError:
        # 如果文件不存在，初始化为空列表
        data = []
    # 确保是列表格式后追加数据
    if isinstance(data, list):
        data.append(new_data)  # 追加数据
    else:
        raise ValueError("JSON 文件的根元素必须是一个列表！")
    # 将更新后的数据重新写入文件
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("写入完成1")

    PURE_INSTRUCT = "You are a helpful agent."
    PURE_INSTRUCT += RESTRICT_INSTRUCT #也加限定
    agent = Agent_gpt(PURE_INSTRUCT, 'agent') 
    final_res = agent.chat(patient_condition) #agent内部需要记录一下message
    final_drug_list = get_drug_list(final_res)
    final_drug_atc_codes = process_drug2atc(final_drug_list)
    print("1111111")
    final_output = {
        "patient_condition": patient_condition,
        "target_drug": {
            "target": list_test_sample["target"],
            "target_drug_code": list_test_sample["drug_code"]
        },
        "final_medicine": { 
            "text": final_res,#文本存档
            "medicine_list": final_drug_list, #药物名称
            "atc3_code_list": final_drug_atc_codes, #
        }
    }
    print("22222222")
    #将结果写入文件
    output_path = "/llm/Code-for-DDI/RAGents4DDI/output/2025-03-19-gpt4o_pure_top_30_mimic3.json"
    # 新的数据
    print("3333333")
    new_data = final_output  
    # 读取现有数据并追加
    try:
        with open(output_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # 读取现有数据，文件必须为标准 JSON 格式
    except FileNotFoundError:
        # 如果文件不存在，初始化为空列表
        data = []
    # 确保是列表格式后追加数据
    if isinstance(data, list):
        data.append(new_data)  # 追加数据
    else:
        raise ValueError("JSON 文件的根元素必须是一个列表！")
    # 将更新后的数据重新写入文件
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("写入完成2")

    return "ok"

# 定义最大重试次数
MAX_RETRIES = 1
# 重试间隔（秒）
RETRY_DELAY = 10
TIMEOUT = 10800  # 超时时间（秒）

def safe_get_medicine_recommendation(vote_times, list_test_sample, top_n):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # 使用 ThreadPoolExecutor 设置超时
            with ThreadPoolExecutor() as executor:
                future = executor.submit(get_medicine_recommendation, vote_times, list_test_sample, top_n)
                result = future.result(timeout=TIMEOUT)  # 设置超时时间

            time.sleep(RETRY_DELAY)  # 延迟一段时间
            return result  # 如果执行成功，则返回结果

        except TimeoutError:
            print(f"函数执行超时（超过 {TIMEOUT} 秒），尝试第 {retries + 1} 次重试...")
        except Exception as e:
            print(f"出错了: {e}")

        retries += 1
        print(f"正在尝试第 {retries} 次重试...")
        time.sleep(RETRY_DELAY)  # 等待一段时间再重试

    print("重试次数超过限制，函数执行失败。")
    return 'None'  # 超过最大重试次数后返回 'None'


# 调用安全函数
unsuccess_item = []
# top_n_list = [10, 20, 25, 30, 40, 50, 60, 70, 80, 90]
# top_n_list = [30]
# for top_n in top_n_list:
top_n = 30
for index, list_test_sample in enumerate(list_test_samples):
    print(index+1,'/',len(list_test_samples))
    #没成功的条目
    vote_times = 0 #记录投票次数--相当于讨论次数
    res = safe_get_medicine_recommendation(vote_times, list_test_sample, top_n)
    if res == "None":
        unsuccess_item.append(index)
    print(unsuccess_item)
    with open(f"/output/unsuccess_gpt4o_0-50_rag_node2_0319.json", "w", encoding="utf-8") as file:
        json.dump(unsuccess_item, file, ensure_ascii=False, indent=4)