import json
import numpy as np
from transformers import RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from transformers import pipeline
from bert_score import score as bert_score



import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'



import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
generation_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')


def generate_response(history, knowledge):

    input_text = history + " " + knowledge

    input_ids = generation_tokenizer.encode(input_text, return_tensors="pt")

    output = generation_model.generate(input_ids, max_length=128, do_sample=True, temperature=0.7)

    response = generation_tokenizer.decode(output[0], skip_special_tokens=True)
    return response







def calculate_scores(response_test, response_gen):
    # Calculate BLEU score
    bleu_score = corpus_bleu([[ref] for ref in response_gen], response_test)
    
    # Calculate ROUGE score
    rouge = Rouge()
    scores = rouge.get_scores(response_test, response_gen, avg=True)
    rouge_score = scores["rouge-l"]["f"]
    
    # Calculate BERTScore
    bert_score_sum = 0.0
    _, _, bert_scores = bert_score(response_gen, response_test, lang='en', model_type='bert-base-uncased', verbose=False)
    for score in bert_scores:
        bert_score_sum += score.item()
    bert_score_avg = bert_score_sum / len(bert_scores)
    
    return bleu_score, rouge_score, bert_score_avg






def load_data(task):

    with open(task + ".json", "r") as f:
        data = json.load(f)

    input_ids = []
    attention_masks = []
    labels = []
    history = []
    knowledge = []
    response = []
    test_response = []
    label = 0
    for i in range(1):  
        for j in range(len(data[i]["utterances"])):
            history.append(' '.join(data[i]["utterances"][j]["history"]))
            knowledge.append(data[i]["utterances"][j]["knowledge"])
            if task == "train":
                response.append(data[i]["utterances"][j]["response"])
            elif task == "test":
                response.append(generate_response(history[-1], knowledge[-1]))
                test_response.append(data[i]["utterances"][j]["response"])
            temp = data[i]["utterances"][j]["BEGIN"]
            x = 0
            
            if data[i]["utterances"][j]["original_response"] is None:
                label = 1
            elif "Hallucination" in temp:
                label = 0
            elif "Entailment" in temp:
                label = 1
            else:
                x = 1
                knowledge.pop()
                response.pop()
                continue
            
            encoded = tokenizer.encode_plus(
                knowledge[-1],
                response[-1],
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(label)

    if task == "train":
        return input_ids, attention_masks, labels
    elif task == "test":
        return input_ids, attention_masks, labels, response, test_response





if __name__ == "__main__":

    # Creating train dataset
    input_ids, attention_masks, labels = load_data("train")
    labels = tf.reshape(labels, [-1, 1])

    train_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, labels))

    train_dataset = train_dataset.map(lambda input_ids, attention_masks, labels:
                                    (tf.cast(input_ids, tf.int32), 
                                    tf.cast(attention_masks, tf.int32), 
                                    labels))


    # Defining the Model
    model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
    

    # Compiling the Model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])




    # Training the model in epoch steps
    batch_size = 16
    num_epochs = 1
    steps = len(labels) // batch_size

    steps=1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step in range(steps):

            low_batch_count = step*batch_size
            high_batch_count = (step+1)*batch_size

            batch_input_ids = input_ids[low_batch_count: high_batch_count]
            batch_attention_masks = attention_masks[low_batch_count: high_batch_count]
            batch_labels = labels[low_batch_count: high_batch_count]

            batch_input_ids = tf.convert_to_tensor(batch_input_ids)
            batch_attention_masks = tf.convert_to_tensor(batch_attention_masks)

            loss, accuracy = model.train_on_batch(x=(batch_input_ids, batch_attention_masks), y=batch_labels)
            if step % 100 == 0:
                print(f"Step {step}/{steps}")
        
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")








    # Creating test dataset

    input_ids, attention_masks, y_test, response, test_response = load_data("test")
    labels = tf.reshape(y_test, [-1, 1])

    test_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, labels))

    test_dataset = test_dataset.map(lambda input_ids, attention_masks, labels:
                                    (tf.cast(input_ids, tf.int32), 
                                    tf.cast(attention_masks, tf.int32), 
                                    labels))



    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)


    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(x=(input_ids, attention_masks), y=labels, verbose=0)

    # Predict output labels
    y_pred = model.predict(x=(input_ids, attention_masks))
    y_pred = np.argmax(y_pred.logits, axis=1)

    # Calculate f1-score
    f1 = f1_score(y_test, y_pred, average='macro')

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)


    print("Test Accuracy:", test_accuracy)
    print("F1-Score:", f1)
    print("Confusion Matrix:")
    print(cm)





    blue_score, rouge_score, bert_score = calculate_scores(test_response, response)

    print("BLUE Score", blue_score)
    print("ROUGE Score", rouge_score)
    print("BERT Score", bert_score)









    # Creating test dataset

    input_ids, attention_masks, y_test = load_data("test")
    labels = tf.reshape(y_test, [-1, 1])

    test_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, labels))

    test_dataset = test_dataset.map(lambda input_ids, attention_masks, labels:
                                    (tf.cast(input_ids, tf.int32), 
                                    tf.cast(attention_masks, tf.int32), 
                                    labels))



    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)


    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(x=(input_ids, attention_masks), y=labels, verbose=0)

    # Predict output labels
    y_pred = model.predict(x=(input_ids, attention_masks))
    y_pred = np.argmax(y_pred.logits, axis=1)

    # Calculate f1-score
    f1 = f1_score(y_test, y_pred, average='macro')

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)


    print("Test Accuracy:", test_accuracy)
    print("F1-Score:", f1)
    print("Confusion Matrix:")
    print(cm)


