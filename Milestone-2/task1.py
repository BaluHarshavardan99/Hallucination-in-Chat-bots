import json
import numpy as np
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, TFRobertaModel, AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Dropout, LSTM, Embedding
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from collections import Counter

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
tokenizer = AutoTokenizer.from_pretrained("roberta-large")


def load_data(task):

    with open(task + ".json", "r") as f:
        data = json.load(f)

    input_ids = []
    attention_masks = []
    labels = []
    knowledge = []
    response = []
    label = 0

    max_res = 0
    len_res = []
    for i in range(len(data)):  
        for j in range(len(data[i]["utterances"])):
            knowledge.append(data[i]["utterances"][j]["knowledge"].lower())

            sentence = data[i]["utterances"][j]["response"].lower()
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            words = sentence.split()
            filtered_words = [word for word in words if word not in stopwords.words('english')]
            response.append(' '.join(filtered_words))


            len_res.append(len(data[i]["utterances"][j]["response"].split()))

            if len(data[i]["utterances"][j]["response"].split()) > max_res:
                max_res = len(data[i]["utterances"][j]["response"].split())
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
                max_length=64,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(label)

    return input_ids, attention_masks, labels



if __name__ == "__main__":


    # Creating train dataset
    input_ids, attention_masks, labels = load_data("train")
    labels = tf.reshape(labels, [-1, 1])

    train_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, labels))

    train_dataset = train_dataset.map(lambda input_ids, attention_masks, labels:
                                    (tf.cast(input_ids, tf.int32), 
                                    tf.cast(attention_masks, tf.int32), 
                                    labels))
    


    input_ids = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="attention_masks")

    # roberta = TFRobertaModel.from_pretrained('roberta-large')
    model = TFAutoModelForSequenceClassification.from_pretrained('roberta-large')

    # for layer in roberta.layers:
    #     layer.trainable = False

    # x = roberta([input_ids, attention_masks])[0]
    # for i in range(4):
    #     self_attention = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=1024)
    #     self_output = tf.keras.layers.Dense(units=1024, activation='relu', name=f'output_{i}')
    #     intermediate = tf.keras.layers.Dense(units=4096, activation='gelu', name=f'intermediate_{i}')
    #     output = tf.keras.layers.Dense(units=1024, activation='relu', name=f'output_{i}')

    #     attention = self_attention(x, x)
    #     attention = tf.keras.layers.Dropout(rate=0.1)(attention)
    #     x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + x)
    #     intermediate_output = intermediate(x)
    #     layer_output = output(intermediate_output)
    #     layer_output = tf.keras.layers.Dropout(rate=0.1)(layer_output)
    #     x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(layer_output + x)
    #     x = tf.keras.layers.Dropout(rate=0.1)(x)

    # x = tf.keras.layers.Flatten()(x)
    # output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)

    # model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output_layer)







    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()





    # define a new model that outputs the output of each layer
    # layer_outputs = [layer.output for layer in model.layers]
    # layer_names = [layer.name for layer in model.layers]
    # layer_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    batch_size = 10
    num_epochs = 1
    steps = len(labels) // batch_size

    steps = 100



    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, (batch_input_ids, batch_attention_masks, batch_labels) in enumerate(train_dataset.batch(batch_size)):

            loss, accuracy = model.train_on_batch(x=(batch_input_ids, batch_attention_masks), y=batch_labels)

            # print(f"Step {step}/{steps}")

            if step % 10 == 0:
                print(f"Step {step}/{steps}")
                x=0
                
                # activations = layer_model.predict((batch_input_ids, batch_attention_masks))
                # for i in range(len(layer_names)):
                #     print(f"{layer_names[i]} output: {activations[i]}")
                
            if step == steps:
                break
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")

    print("Training Complete")


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



    y_pred = model.predict(x=(input_ids, attention_masks))
    y_pred = np.argmax(y_pred.logits, axis=1)



    # kk=[]
    # for i in range(len(y_test)):
    #     x = y_pred[i]
    #     kk.append("{} {}".format(y_test[i], np.mean(x)))
    # print(Counter(kk))


    # temp = []
    # ch = max(np.median(y_pred), np.mean(y_pred))
    # for x in y_pred:
    #     if x <= ch:
    #         temp.append(0)
    #     else:
    #         temp.append(1)

    # y_pred = temp


    # kk=[]
    # for i in range(len(y_test)):
    #     kk.append("{} {}".format(y_test[i], y_pred[i]))
    # print(Counter(kk))

    print(y_test)
    print(y_pred)

    # Calculate f1-score
    f1 = f1_score(y_test, y_pred, average='macro')

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # print("Test Accuracy:", test_accuracy)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-Score:", f1)
    print("Confusion Matrix:")
    print(cm)
