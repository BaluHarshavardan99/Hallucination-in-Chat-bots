# Hallucination in Chat-bots: Faithful Benchmark for Information-Seeking Dialogue - Fact Hallucinations Detection and Prevention

This repository contains the code for the final project of the course "CSE 635: Natural Language Processing and Information Retrieval." The project focuses on detecting and preventing hallucinations in Information-Seeking Dialogue-based NLP systems.

## Abstract
Chatbots offer a convenient means of communication but are susceptible to generating factually incorrect responses, known as hallucinations. These errors can mislead users and damage the chatbot's credibility. In this project, we propose a solution to detect and mitigate hallucinations in chatbot outputs, particularly in Information-Seeking Dialogues.

We leverage both rule-based techniques and machine learning models to identify and flag hallucinated content. Our approach involves:
- Detecting hallucinations using a combination of rule-based methods and RoBERTa-large, a pre-trained language model.
- Preventing future hallucinations by modifying the training data, reweighting model parameters, and incorporating feedback loops with human experts.

The ultimate goal of this project is to improve user trust, enhance the reliability of chatbots, and reduce the harm caused by misinformation.

## Model Overview

1. Baseline Model (milestone-2)
   - Uses RoBERTa-large as the foundation.
   - Implemented in `milestone-2/task1.py`.

2. Final Model (milestone-3)
   - A modified version of RoBERTa-large with additional techniques for hallucination prevention.


## Instructions to Run the Code

This repository contains two folders: `milestone-2` and `milestone-3`.

### 1. Baseline Model (milestone-2)
The baseline model is implemented using the **RoBERTa-large** architecture.

To train and test the baseline model, follow these steps:
```bash
python task1.py
``` 




## Tasks

### Task 1: Training and Testing

To run the first task, execute:

```bash
python task1.py
```

* This command will train the model and save the resulting model as `task1_model.pth`.
* If you only want to test the model, comment out the `trainer.fit()` and `torch.save()` lines in the code.

### Task 2: BEGIN and VRM Classification

To run the classification tasks, use:

```bash
python task2.py
```

* This will train and test the models for BEGIN and VRM classification.
* The trained models will be saved as `task21_model.pth` and `task22_model.pth`, respectively.
* If you only want to test the models, comment out the `trainer.fit()` and `torch.save()` lines in the code.

### Task 3: Hallucination Classification

Before running this task, ensure that the `task1_model.pth` is generated by running Task 1.

To classify hallucinations, run:

```bash
python task3.py
```

* This task uses the model trained in Task 1 to classify whether the chatbot's responses are hallucinated or not.

## Repository Structure

```
├── milestone-2
│   ├── task1.py    # Baseline model code using RoBERTa-large
│   ├── utils.py    # Utility functions
├── milestone-3
│   ├── task1.py    # Final model code for task 1
│   ├── task2.py    # Code for BEGIN and VRM classification
│   ├── task3.py    # Hallucination classification code
│   ├── utils.py    # Utility functions for final model
├── requirements.txt # Dependencies for the project
└── README.md       # Project description and instructions
```

## Requirements

* Python 3.x
* PyTorch
* Transformers
  



## Future Improvements

* Integrating more sophisticated techniques to reduce hallucinations, such as reinforcement learning with human feedback.
* Exploring additional datasets for broader generalization.
* Refining rule-based techniques for more accurate detection.





