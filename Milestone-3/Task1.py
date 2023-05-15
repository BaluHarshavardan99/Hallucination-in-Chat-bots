import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AdamW, AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import sys
from pathlib import Path
import datasets
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())


# setting batch size
BATCH_SIZE = 10

# number of training epochs
EPOCHS = 5


# definining number of classes
NUM_CLASSES = 2


# Processor to process data
class TaskProcessor:

    # Initializing
    def __init__(self, **kwargs):
        pass

    # loading data as json
    def load(self, fold: str):

        examples = []
        # loading data from huggingface datasets
        dataset = datasets.load_dataset("McGill-NLP/FaithDial", split=fold)

        # Iterating dataset to get conversations
        for i, item in enumerate(dataset):
   
            # Coding labels
            label = item["BEGIN"]
            if "Entailment" in label:
                item["label_id"] = 0
            elif "Hallucination" in label:
                item["label_id"] = 1
            else:
                item["label_id"] = 0

            # Labels = Entailment or Hallucination
            # Label for Entailment = 0
            # Label for Hallucination = 0


            # retrieving knowledge and response
            knowledge = item["knowledge"].strip()
            original_response = item["original_response"]
            response = item["response"]

            # adding label id
            # item["label_id"] = 1 if label == "Entailment" else 0


            # returning examples
            examples.append(item)

            # for when original response
            if fold == "train":
                if original_response or response != original_response:
                    examples.append({
                        "guid": f"{fold}-{i}",
                        "knowledge": knowledge,
                        "response": response,
                        "label": "Entailment",
                        "label_id": 1
                    })

        # returning single examples
        return examples




# Class for retrieving training and testing datasets and processing
class TaskDataModule(pl.LightningDataModule):
    # Initializing
    def __init__(self, tokenizer):
        super().__init__()
        # retrieving roberta tokenizer
        self.tokenizer = tokenizer
        # initializing task processor to preprocess data
        self.processor = TaskProcessor()

    # code to setup train and test and encode data
    def setup(self, stage: Optional[str] = None):

        # initializing data json
        data = {}

        # adding train data
        data["train"] = self.processor.load("train")
        # data["train"] = data["train"][:1000]

        # adding test data
        data["test"] = self.processor.load("test")
        # data["test"] = data["test"][:1000]

        # started encoding to input ids and attention masks

        # encoding for training
        encoded = []
        for ex in data["train"]:
            temp = self.tokenizer(ex["knowledge"], ex["response"], truncation=True)
            temp["label"] = ex["label_id"]
            encoded.append(temp)
            # returns ids, masks and labels
        data["train"] = encoded

        # encoding test data
        encoded = []
        for ex in data["test"]:
            temp = self.tokenizer(ex["knowledge"], ex["response"], truncation=True)
            temp["label"] = ex["label_id"]
            encoded.append(temp)
            # returns ids, masks and labels
        data["test"] = encoded

        # passing data to dataset
        self.dataset = data
    
    # train loader inbuild function
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=BATCH_SIZE, collate_fn=DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=4))

    # test loader inbuilt function
    def test_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(self.dataset["test"], batch_size=BATCH_SIZE, collate_fn=DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=4),)






# Trainer Class
class model_build(pl.LightningModule):
    def __init__(
        self,
        # hyperparameters
        hparams: argparse.Namespace,
    ):
        super().__init__()

        # init processor
        self.processor = TaskProcessor()

        # 2 classes: entailment and hallucination
        self.num_classes = NUM_CLASSES

        # defining roberta large tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # defining roberta large model
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large",)


    # Forward pass the model
    def forward(self, input_ids, attention_mask, token_type_ids):
        print(token_type_ids)
        # returning trained model
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,)

    # configuring optimizers
    def configure_optimizers(self):

        # setting model
        model = self.model

        # init parameters for layer training
        par_1 = []
        par_2 = []
        for n, a in model.named_parameters():
            # these layers no change in weight decay
            if any(nd in n for nd in ["LayerNorm.weight", "bias", "embedding.weight", "position_embeddings.weight"]):
                par_1.append(a)
            else:
                par_2.append(a)


        # setting parameters
        parmets = [{"params": par_1, "weight_decay": 0.0}, {"params": par_2, "weight_decay": 0.1}]

        # setting adam optimizer
        optimizer = AdamW(params = parmets, lr = 1e-5, eps = 1e-8)
        
        # setting steps
        steps = (self.dataset_size / BATCH_SIZE) * EPOCHS
    
        # defining scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= steps//20, num_training_steps= steps)
        scheduler = {"scheduler": scheduler, "frequency": 1, "interval": "step"}

        # returning optimizer and scheduler
        return [optimizer], [scheduler]


    # training setps definied for inbuilt function
    def training_step(self, batch, batch_idx):
        # get batch inpus
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # get batch labels
        labels = batch['labels']
        
        # get training loss outputs
        # output logits are generated
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        # getting loss
        loss = outputs.loss

        # return training loss
        return loss
    
    
    # test setps definied for inbuilt function
    def test_step(self, batch, batch_idx):
        # get batch input and masks
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # get y_test labels
        labels = batch['labels']
        
        # get predicted labels
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # retrieve labels and predictions for metrics
        y_test = labels.cpu().numpy()
        y_pred = outputs.logits.argmax(dim=-1).cpu().numpy()

        # labels = y_test
        # predictions = y_pred
            
        # returning loss and test and preds
        return {'loss': loss, 'y_test': y_test, 'y_pred': y_pred}
    

    # defining last test phase to print metrics
    def test_epoch_end(self, outputs):
        # getting y_test
        y_test = [x['y_test'] for x in outputs]
        y_test = np.concatenate(y_test)

        # getting y_pred
        y_pred = [x['y_pred'] for x in outputs]
        y_pred = np.concatenate(y_pred)

        # printing accuracy
        ac = accuracy_score(y_test, y_pred)
        #printing f1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        # printing confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print("Test Accuracy:", ac)
        print("Test F1 Score:", f1)
        print("Confusion Matrix:\n", cm)
    

    # code for model setup
    def setup(self, stage):
        # if setup is to fit get train data size
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()
            # get dataset size
            ds_size = len(train_loader.dataset)
            self.dataset_size = ds_size





# Main Code starts
if __name__ == "__main__":
    
    # Initializing Model
    args = argparse.ArgumentParser().parse_args()
    model = model_build(args)
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, max_epochs = EPOCHS)


    # For Training

    trainer.fit(model, datamodule = TaskDataModule(model.tokenizer))
    torch.save(model.state_dict(), 'task1_model.pth')


    # For Testing

    model.load_state_dict(torch.load('best_model.pth'))
    trainer.test(model, datamodule=TaskDataModule(model.tokenizer))
