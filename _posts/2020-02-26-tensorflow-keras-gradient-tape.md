# TensorFlow GradientTape

How (not) to train your functional Keras model!

Let's say we want to train an NLP model to maximize similarity between two inputs : X1 and X2.
```python
import tensorflow as tf
import numpy as np

model = some_model()
y1 = model(X1)
y2 = model(X2)
```

Say, for example, we want to maximize the similarity between y1 and y2. The loss function for this looks like:
```python
cosine_loss = tf.keras.losses.CosineSimilarity()
```

Now, let's define and train such a model using `GradientTape`.





1. TOC
{:toc}

## Section 1 : Define Model 
```python

# constants
vocab_size, embedding_dim = 1000, 100
batch_size = 16
max_len = 10
lstm_hidden_size = 64
recurrent_dropout = 0.1
return_sequences = False

# functional model layers
emb_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
lstm_layer = tf.keras.layers.LSTM(
    lstm_hidden_size, 
    recurrent_dropout=recurrent_dropout,
    return_sequences=return_sequences
)
def model(x):
    x = emb_layer(x)
    x = lstm_layer(x)
    return x
```
We will be using this simple NLP model : Embedding layer followed by LSTM layer. 

## Section 2 : How not to train your functional keras model using `GradientTape`

### The first wrong way: 
The first wrong of training the model involves calculating the output _outside_ of the GradientTape **context**:
```python
# example inputs, for illustration only.
X1 = np.random.randint(0, vocab_size, (batch_size, max_len))
X1 = np.random.randint(0, vocab_size, (batch_size, max_len))

# calculating model outputs outside of the GradientTape context
y1 = model(X1)
y2 = model(X2)
with tf.GradientTape() as tape:
    loss = cosine_loss(y1, y2)
variables = model.trainable_variables
gradients = tape.gradient(loss, variables)
optimizer.apply_gradients(zip(gradients, variables))
```
If you try to run this block of code, it will give you following error:
```
ValueError: No gradients provided for any variable: .... 
```
This is because the GradientTape context has not tracked the way `y1` and `y2` were obtained, hence it does not have access to variables which were used to obtain `y1` and `y2`. Hence the error.


### The second wrong way:
The second wrong way to do it is to calculate the loss _outside_ of the GradientTape **context**:
```python
X1 = np.random.randint(0, vocab_size, (batch_size, max_len))
X1 = np.random.randint(0, vocab_size, (batch_size, max_len))

with tf.GradientTape() as tape:
    y1 = model(X1)
    y2 = model(X2)
# calculating the loss outside of the GradientTape context
loss = cosine_loss(y1, y2)
variables = model.trainable_variables
gradients = tape.gradient(loss, variables)
optimizer.apply_gradients(zip(gradients, variables))
```

If you try to run this block of code, it will give you the same error:
```
ValueError: No gradients provided for any variable: .... 
```
This is because the GradientTape context has not tracked the way `loss` was calculated, does not have access to variables needed to calculate `loss` and could not backpropage it. Hence the error.


## Section 3 : How to train your functional keras model using `GradientTape` _the right way_
The right way to train a model is let the `GradientTape` context track both the output calculation and loss calculation! This took a bit of a learning for me.
```python
X1 = np.random.randint(0, vocab_size, (batch_size, max_len))
X1 = np.random.randint(0, vocab_size, (batch_size, max_len))

# calculate outputs and loss within the `GradientTape` context
with tf.GradientTape() as tape:
    y1 = model(X1)
    y2 = model(X2)
    loss = cosine_loss(y1, y2)
variables = model.trainable_variables
gradients = tape.gradient(loss, variables)
optimizer.apply_gradients(zip(gradients, variables))
```
This block of code will run without a hiccup and will also train your models (as a side effect :P)


# Lesson:

When using functional models from `tf.keras`, track your calculations of all the ops needed to compute the loss _with_ the GradientTape context!  

