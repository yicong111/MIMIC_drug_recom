## 💊 Experiences on Drug Recommendation through RAG-enhanced LLM-based Multi-Agents

This project explores how **Retrieval-Augmented Generation (RAG)** and **multi-agent collaboration based on Large Language Models (LLMs)** can be leveraged to generate safer, more interpretable drug recommendations — with a particular focus on minimizing **drug-drug interactions (DDIs)**.

---

### 📦 Code Guide

1. **Data Preparation**
   - Data preprocessing follows the pipeline from *"Large Language Model Distilling Medication Recommendation Model"* (`construction.ipynb`).
   - Drug interaction information and ATC code conversion are based on the [GAMENet](https://github.com/sjy1203/GAMENet) dataset and logic.

2. **RAG Index Creation**
   - After preprocessing, install the required dependencies and run:
     ```bash
     python create_index_ddi_info.py
     ```
   - This creates the DDI knowledge index for use in RAG.

3. **Model Configuration**
   - Insert your API key in `utils.py`, or configure a local LLM endpoint if self-hosting.

4. **Run the Multi-Agent Simulation**
   ```bash
   python node_experiment_rag.py

---

### 🧠 Key Findings 
1. **Initial Setup**
    The RAG used a MedCPT+MedCorpus index (textbooks, PubMed, Wikipedia, StatPearls), but retrieving irrelevant content actually worsened recommendation quality.
2. **Refinements**
    Added MIMIC-derived real-world cases and drug conflict data — but performance still did not improve.
    Found that limiting the candidate drug set was essential to avoid ATC code conversion errors.
    Introduced atc2drug and drugbank indices to refine ATC-level mapping.
    ➖ Minimalist RAG:
    Performance improved when we reduced the RAG index to only DDI-related content — leading to a significant DDI reduction, comparative to SOTA.
3. **Ablation & Analysis**
    Ablation studies confirmed RAG was necessary.
    Re-editing the prescription after one round of conflict resolution did not further reduce DDI.
    Compared single decision vs. multi voting: single decisions performed better.
    Adding drug usage frequency-related prompts boosted F1 scores by guiding more realistic drug usage decisions.
4. **⚠️ Challenges Observed**
    Some models generated excessive prescriptions due to weak reasoning ability.
    In longer sessions, context overflow led to performance drops; resolved by pruning unnecessary dialogue rounds.
5. **🔚 Final Insights**
    While DDI reduction was successful, the generated prescriptions did not fully meet all clinical needs — due to prioritization of conflict resolution over treatment completeness.
    Overall, this project demonstrates a practical, interpretable approach to drug recommendation through RAG-enhanced LLM-based multi-agent reasoning, achieving state-of-the-art performance in DDI mitigation.