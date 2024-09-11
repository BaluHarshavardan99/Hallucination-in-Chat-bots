# Hallucination in Chat-bots: Faithful Benchmark for Information-Seeking Dialogue - Fact Hallucinations Detection and Prevention

This repository contains the code for the final project of the course "CSE 635: Natural Language Processing and Information Retrieval." The project focuses on detecting and preventing hallucinations in Information-Seeking Dialogue-based NLP systems.

## Abstract
Chatbots offer a convenient means of communication but are susceptible to generating factually incorrect responses, known as hallucinations. These errors can mislead users and damage the chatbot's credibility. In this project, we propose a solution to detect and mitigate hallucinations in chatbot outputs, particularly in Information-Seeking Dialogues.

We leverage both rule-based techniques and machine learning models to identify and flag hallucinated content. Our approach involves:
- Detecting hallucinations using a combination of rule-based methods and RoBERTa-large, a pre-trained language model.
- Preventing future hallucinations by modifying the training data, reweighting model parameters, and incorporating feedback loops with human experts.

The ultimate goal of this project is to improve user trust, enhance the reliability of chatbots, and reduce the harm caused by misinformation.

## Instructions to Run the Code

This repository contains two folders: `milestone-2` and `milestone-3`.

### 1. Baseline Model (milestone-2)
The baseline model is implemented using the **RoBERTa-large** architecture.

To train and test the baseline model, follow these steps:
```bash
python task1.py



