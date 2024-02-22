This folder contains milestone-2 code and detailed instructions to run the code 

**Instructions to run the code:**
1) For task-1: "python task1.py" - it will train and test the model. And after training, it will save the generated model as task1_model.pth. If you want to just test the model, you can comment out trainer.fit and torch.save lines from the code.
2) For task-2: 
   BEGIN Classification: "python task2_vrm.py" - it will train and test the model
   VRM Classification: "python task2_begin.py" - it will train and test the model 
4) For task-3: "python task3.py" - you need to run task-1 before you run task-3 so that the task1 model .pth is generated which will then be used to classify whether the generated responses are hallucinated or not.
