import json
import numpy as np
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'



tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def load_data(task):
    with open(task + ".json", "r") as f:
        data = json.load(f)

    input_ids = []
    attention_masks = []
    labels = []
    for i in range(len(data)):
        for j in range(len(data[i]["utterances"])):
            label = np.zeros((4,), dtype=np.int32)
            for k in range(len(data[i]["utterances"][j]["BEGIN"])):
                if data[i]["utterances"][j]["BEGIN"][k] == 'Hallucination':
                    label[0] = 1
                elif data[i]["utterances"][j]["BEGIN"][k] == 'Entailment':
                    label[1] = 1
                elif data[i]["utterances"][j]["BEGIN"][k] == 'Uncooperative':
                    label[2] = 1
                elif data[i]["utterances"][j]["BEGIN"][k] == 'Generic':
                    label[3] = 1

            encoded = tokenizer.encode_plus(
                data[i]["utterances"][j]["knowledge"],
                data[i]["utterances"][j]["response"],
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(label)

    return input_ids, attention_masks, np.array(labels)

if __name__ == "__main__":
    # Creating train dataset
    input_ids, attention_masks, labels = load_data("train")
    train_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, labels))
    train_dataset = train_dataset.map(lambda input_ids, attention_masks, labels:
                                      (tf.cast(input_ids, tf.int32), 
                                       tf.cast(attention_masks, tf.int32), 
                                       tf.cast(labels, tf.int32)))

    # Defining the Model
    model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4, output_attentions=False, output_hidden_states=False)

    # Compiling the Model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    # Training the model
    batch_size = 16
    num_epochs = 1
    steps = len(labels) // batch_size

    steps = 9

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
        
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")



   # Creating test dataset
    input_ids, attention_masks, y_test = load_data("test")
    labels = tf.convert_to_tensor(y_test)

    test_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, labels))

    test_dataset = test_dataset.map(lambda input_ids, attention_masks, labels:
                                    (tf.cast(input_ids, tf.int32), 
                                    tf.cast(attention_masks, tf.int32), 
                                    tf.cast(labels, tf.int32)))


    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)


    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(x=(input_ids, attention_masks), y=labels, verbose=0)

    # Predict output labels
    y_pred = model.predict(x=(input_ids, attention_masks))


    y_pred = tf.math.round(y_pred.logits)
    y_pred = np.round(y_pred)
    y_pred = tf.where(y_pred < 0, 0, y_pred)

    


    # Calculate f1-score

    try:
        f1 = f1_score(y_test, y_pred.numpy(), average='macro')
    except:
        f1 = f1_score(y_test, y_pred, average='macro')

    print("Test Accuracy:", test_accuracy)
    print("F1-Score:", f1)



    cm = multilabel_confusion_matrix(y_test, y_pred.numpy())

    print("Confusion Matrix:")
    for i, label in enumerate(['Hallucination', 'Entailment', 'Uncooperative', 'Generic']):
        print("Label: ", label)
        print(cm[i])

