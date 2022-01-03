# Inspect the first sentence on `X_test`
print(X_test[0])

# Get the predicion for all the sentences
pred = model.predict(X_test)

# Transform the predition into positive (> 0.5) or negative (<= 0.5)
pred_sentiment = ["positive" if x>0.5 else "negative" for x in pred]

# Create a data frame with sentences, predictions and true values
result = pd.DataFrame({'sentence': sentences, 'y_pred': pred_sentiment, 'y_true': y_test})

# Print the first lines of the data frame
print(result.head())

==========================================

# Transform the list of sentences into a list of words
all_words = ' '.join(sheldon_quotes).split(' ')

# Get number of unique words
unique_words = list(set(all_words))

# Dictionary of indexes as keys and words as values
index_to_word = {i:wd for i, wd in enumerate(sorted(unique_words))}

print(index_to_word)

# Dictionary of words as keys and indexes as values
word_to_index = {wd:i for i, wd in enumerate(sorted(unique_words))}

print(word_to_index)

===========================================

# Create lists to keep the sentences and the next character
sentences = []   # ~ Training data
next_chars = []  # ~ Training labels

# Define hyperparameters
step = 2          # ~ Step to take when reading the texts in characters
chars_window = 10 # ~ Number of characters to use to predict the next one  

# Loop over the text: length `chars_window` per time with step equal to `step`
for i in range(0, len(sheldon_quotes) - chars_window, step):
    sentences.append(sheldon_quotes[i:i + chars_window])
    next_chars.append(sheldon_quotes[i + chars_window])

# Print 10 pairs
print_examples(sentences, next_chars, 10)

===========================================

# Loop through the sentences and get indexes
new_text_split = []
for sentence in new_text:
    sent_split = []
    for wd in sentence.split(' '):
        index = word_to_index.get(wd, 0)
        sent_split.append(index)
    new_text_split.append(sent_split)

# Print the first sentence's indexes
print(new_text_split[0])

# Print the sentence converted using the dictionary
print(' '.join([index_to_word[index] for index in new_text_split[0]]))

===========================================

# Instantiate the class
model = Sequential(name="sequential_model")

# One LSTM layer (defining the input shape because it is the 
# initial layer)
model.add(LSTM(128, input_shape=(None, 10), name="LSTM"))

# Add a dense layer with one unit
model.add(Dense(1, activation="sigmoid", name="output"))

# The summary shows the layers and the number of parameters 
# that will be trained
model.summary()

=============================================

# Define the input layer
main_input = Input(shape=(None, 10), name="input")

# One LSTM layer (input shape is already defined)
lstm_layer = LSTM(128, name="LSTM")(main_input)

# Add a dense layer with one unit
main_output = Dense(1, activation="sigmoid", name="output")(lstm_layer)

# Instantiate the class at the end
model = Model(inputs=main_input, outputs=main_output, name="modelclass_model")

# Same amount of parameters to train as before (71,297)
model.summary()

==============================================

# Import relevant classes/functions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Build the dictionary of indexes
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Change texts into sequence of indexes
texts_numeric = tokenizer.texts_to_sequences(texts)
print("Number of words in the sample texts: ({0}, {1})".format(len(texts_numeric[0]), len(texts_numeric[1])))

# Pad the sequences
texts_pad = pad_sequences(texts_numeric, 60)
print("Now the texts have fixed length: 60. Let's see the first one: \n{0}".format(texts_pad[0]))

================================================

# Build model
model = Sequential()
model.add(SimpleRNN(units=128, input_shape=(None, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# Load pre-trained weights
model.load_weights('model_weights.h5')

# Method '.evaluate()' shows the loss and accuracy
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Loss: {0} \nAccuracy: {1}".format(loss, acc))

==================================================

# Create a Keras model with one hidden Dense layer
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer=he_uniform(seed=42)))
model.add(Dense(1, activation='linear'))

# Compile and fit the model
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# See Mean Square Error for train and test data
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)

# Print the values of MSE
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

=================================================

# Create a Keras model with one hidden Dense layer
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer=he_uniform(seed=42)))
model.add(Dense(1, activation='linear'))

# Compile and fit the model
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, clipvalue=3.0))
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# See Mean Square Error for train and test data
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)

# Print the values of MSE
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

==================================================

# Create the model
model = Sequential()
model.add(SimpleRNN(units=600, input_shape=(None, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Load pre-trained weights
model.load_weights('model_weights.h5')

# Plot the accuracy x epoch graph
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'val'], loc='upper left')
plt.show()

=================================================

# Import the modules
from keras.layers import GRU, Dense

# Print the old and new model summaries
SimpleRNN_model.summary()
gru_model.summary()

# Evaluate the models' performance (ignore the loss value)
_, acc_simpleRNN = SimpleRNN_model.evaluate(X_test, y_test, verbose=0)
_, acc_GRU = gru_model.evaluate(X_test, y_test, verbose=0)

# Print the results
print("SimpleRNN model's accuracy:\t{0}".format(acc_simpleRNN))
print("GRU model's accuracy:\t{0}".format(acc_GRU))

===================================================

# Import the LSTM layer
from keras.layers.recurrent import LSTM

# Build model
model = Sequential()
model.add(LSTM(units=128, input_shape=(None, 1), return_sequences=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load pre-trained weights
model.load_weights('lstm_stack_model_weights.h5')

print("Loss: %0.04f\nAccuracy: %0.04f" % tuple(model.evaluate(X_test, y_test, verbose=0)))

====================================================

# Import the embedding layer
from keras.layers import Embedding

# Create a model with embeddings
model = Sequential(name="emb_model")
model.add(Embedding(input_dim=80002, output_dim=wordvec_dim, input_length=200, trainable=True))
model.add(GRU(128))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the summaries of the one-hot model
model_onehot.summary()

# Print the summaries of the model with embeddings
model.summary()

======================================================

# Load the glove pre-trained vectors
glove_matrix = load_glove('glove_200d.zip')

# Create a model with embeddings
model = Sequential(name="emb_model")
model.add(Embedding(input_dim=vocabulary_size + 1, output_dim=wordvec_dim, 
                    embeddings_initializer=Constant(glove_matrix), 
                    input_length=sentence_len, trainable=False))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the summaries of the model with embeddings
model.summary()

=========================================================

# Create the model with embedding
model = Sequential(name="emb_model")
model.add(Embedding(input_dim=max_vocabulary, output_dim=wordvec_dim, input_length=max_len))
model.add(SimpleRNN(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load pre-trained weights
model.load_weights('embedding_model_weights.h5')

# Evaluate the models' performance (ignore the loss value)
_, acc_embeddings = model.evaluate(X_test, y_test, verbose=0)

# Print the results
print("SimpleRNN model's accuracy:\t{0}\nEmbeddings model's accuracy:\t{1}".format(acc_simpleRNN, acc_embeddings))

============================================================

# Build and compile the model
model = Sequential()
model.add(Embedding(vocabulary_size, wordvec_dim, trainable=True, input_length=max_text_len))
model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))
model.add(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.15))
model.add(Dense(16))
model.add(Dropout(rate=0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load pre-trained weights
model.load_weights('model_weights.h5')

# Print the obtained loss and accuracy
print("Loss: {0}\nAccuracy: {1}".format(*model.evaluate(X_test, y_test, verbose=0)))

==============================================================

# Print the model summary
model_cnn.summary()

# Load pre-trained weights
model_cnn.load_weights('model_weights.h5')

# Evaluate the model to get the loss and accuracy values
loss, acc = model_cnn.evaluate(x_test, y_test, verbose=0)

# Print the loss and accuracy obtained
print("Loss: {0}\nAccuracy: {1}".format(loss, acc))

================================================================

# Get the numerical ids of column label
numerical_ids = df.label.cat.codes

# Print initial shape
print(numerical_ids.shape)

# One-hot encode the indexes
Y = to_categorical(numerical_ids)

# Check the new shape of the variable
print(Y.shape)

# Print the first 5 rows
print(Y[:5])

===================================================================

# Create and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_dataset.data)

# Prepare the data
prep_data = tokenizer.texts_to_sequences(news_dataset.data)
prep_data = pad_sequences(prep_data, maxlen=200)
print(prep_data)

# Prepare the labels
prep_labels = to_categorical(news_dataset.target)

# Print the shapes
print(prep_data.shape)
print(prep_labels.shape)

=====================================================================

# Import plotting package
import matplotlib.pyplot as plt

# Insert lists of accuracy obtained on the validation set
plt.plot(history_no_emb['acc'], marker='o')
plt.plot(history_emb['acc'], marker='o')

# Add extra descriptions to plot
plt.title('Learning with and without pre-trained embedding vectors')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['no_embeddings', 'with_embeddings'], loc='upper left')

# Display the plot
plt.show()

======================================================================

# Word2Vec model
w2v_model = Word2Vec.load('bigbang_word2vec.model')

# Selected words to check similarities
words_of_interest = ["bazinga", "penny", "universe", "spock", "brain"]

# Compute top 5 similar words for each of the words of interest
top5_similar_words = []
for word in words_of_interest:
    top5_similar_words.append(
      {word: [item[0] for item in w2v_model.wv.most_similar([word], topn=5)]}
    )

# Print the similar words
print(top5_similar_words)

==========================================================================

# See example article
print(news_dataset.data[5])

# Transform the text into numerical indexes
news_num_indices = tokenizer.texts_to_sequences(news_dataset.data)

# Print the transformed example article
print(news_num_indices[5])

# Transform the labels into one-hot encoded vectors
labels_onehot = to_categorical(news_dataset.target)

# Check before and after for the sample article
print("Before: {0}\nAfter: {1}".format(news_dataset.target[5], labels_onehot[5]))

=========================================================================

# Change text for numerical ids and pad
X_novel = tokenizer.texts_to_sequences(news_novel.data)
X_novel = pad_sequences(X_novel, maxlen=400)

# One-hot encode the labels
Y_novel = to_categorical(news_novel.target)

# Load the model pre-trained weights
model.load_weights('classify_news_weights.h5')

# Evaluate the model on the new dataset
loss, acc = model.evaluate(X_novel, Y_novel, batch_size=64)

# Print the loss and accuracy obtained
print("Loss:\t{0}\nAccuracy:\t{1}".format(loss, acc))

=======================================================================

# Get probabilities for each class
pred_probabilities = model.predict_proba(X_test)

# Thresholds at 0.5 and 0.8
y_pred_50 = [np.argmax(x) if np.max(x) >= 0.5 else DEFAULT_CLASS for x in pred_probabilities]
y_pred_80 = [np.argmax(x) if np.max(x) >= 0.8 else DEFAULT_CLASS for x in pred_probabilities]

trade_off = pd.DataFrame({
    'Precision_50': precision_score(y_true, y_pred_50, average=None), 
    'Precision_80': precision_score(y_true, y_pred_80, average=None), 
    'Recall_50': recall_score(y_true, y_pred_50, average=None), 
    'Recall_80': recall_score(y_true, y_pred_80, average=None)}, 
  index=['Class 1', 'Class 2', 'Class 3'])

print(trade_off)

=========================================================================

# Compute the precision of the sentiment model
prec_sentiment = precision_score(sentiment_y_true, sentiment_y_pred, average=None)
print(prec_sentiment)

# Compute the recall of the sentiment model
rec_sentiment = recall_score(sentiment_y_true, sentiment_y_pred, average=None)
print(rec_sentiment)

====================================

# Use the model to predict on new data
predicted = model.predict(X_test)

# Choose the class with higher probability 
y_pred = np.argmax(predicted, axis=1)

# Compute and print the confusion matrix
print(confusion_matrix(y_true, y_pred))

# Create the performance report
print(classification_report(y_true, y_pred, target_names=news_cat))

====================================

# Context for Sheldon phrase
sheldon_context = "I’m not insane, my mother had me tested. "

# Generate one Sheldon phrase
sheldon_phrase = generate_sheldon_phrase(sheldon_model, sheldon_context)

# Print the phrase
print(sheldon_phrase)

# Context for poem
poem_context = "May thy beauty forever remain"

# Print the poem
print(generate_poem(poem_model, poem_context))

====================================

# Transform text into sequence of indexes and pad
X = encode_sequences(sentences)

# Print the sequences of indexes
print(X)

# Translate the sentences
translated = translate_many(model, X)

# Create pandas DataFrame with original and translated
df = pd.DataFrame({'Original': sentences, 'Translated': translated})

# Print the DataFrame
print(df)

====================================

def get_next_char(model, initial_text, chars_window, char_to_index, index_to_char):
  	# Initialize the X vector with zeros
    X = initialize_X(initial_text, chars_window, char_to_index)
    
    # Get next character using the model
    next_char = predict_next_char(model, X, index_to_char)
	
    return next_char

# Define context sentence and print the generated text
initial_text = "I am not insane, "
print("Next character: {0}".format(get_next_char(model, initial_text, 20, char_to_index, index_to_char)))

=====================================

def generate_phrase(model, initial_text):
    # Initialize variables  
    res, seq, counter, next_char = initialize_params(initial_text)
    
    # Loop until stop conditions are met
    while counter < 100 and next_char != r'.':
      	# Get next char using the model and append to the sentence
        next_char, res, seq = get_next_token(model, res, seq)
        # Update the counter
        counter = counter + 1
    return res
  
# Create a phrase
print(generate_phrase(model, "I am not insane, "))

========================================

# Define the initial text
initial_text = "Spock and me "

# Define a vector with temperature values
temperatures = [0.2, 0.8, 1.0, 3.0, 10.0]

# Loop over temperatures and generate phrases
for temperature in temperatures:
	# Generate a phrase
	phrase = generate_phrase(model, initial_text, temperature)
    
	# Print the phrase
	print('Temperature {0}: {1}'.format(temperature, phrase))

========================================

# Instantiate the vectors
sentences = []
next_chars = []
# Loop for every sentence
for sentence in sheldon.split('.'):
    # Get 20 previous chars and next char; then shift by step
    for i in range(0, len(sentence) - chars_window, step):
        sentences.append(sentence[i:i + chars_window])
        next_chars.append(sentence[i + chars_window])

# Define a Data Frame with the vectors
df = pd.DataFrame({'sentence': sentences, 'next_char': next_chars})

# Print the initial rows
print(df.head())

=========================================

# Instantiate the variables with zeros
numerical_sentences = np.zeros((num_seqs, chars_window, n_vocab), dtype=np.bool)
numerical_next_chars = np.zeros((num_seqs, n_vocab), dtype=np.bool)

# Loop for every sentence
for i, sentence in enumerate(sentences):
  # Loop for every character in sentence
  for t, char in enumerate(sentence):
    # Set position of the character to 1
    numerical_sentences[i, t, char_to_index[char]] = 1
    # Set next character to 1
    numerical_next_chars[i, char_to_index[next_chars[i]]] = 1

# Print the first position of each
print(numerical_sentences[0], numerical_next_chars[0], sep="\n")

======================================

# Instantiate the model
model = Sequential(name="LSTM model")

# Add two LSTM layers
model.add(LSTM(64, input_shape=input_shape, dropout=0.15, recurrent_dropout=0.15, return_sequences=True, name="Input_layer"))
model.add(LSTM(64, dropout=0.15, recurrent_dropout=0.15, return_sequences=False, name="LSTM_hidden"))

# Add the output layer
model.add(Dense(n_vocab, activation='softmax', name="Output_layer"))

# Compile and load weights
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights('model_weights.h5')
# Summary
model.summary()

=======================================

# Get maximum length of the sentences
pt_length = max([len(sentence.split(' ')) for sentence in pt_sentences])

# Transform text to sequence of numerical indexes
X = input_tokenizer.texts_to_sequences(pt_sentences)

# Pad the sequences
X = pad_sequences(X, maxlen=pt_length, padding='post')

# Print first sentence
print(pt_sentences[0])

# Print transformed sentence
print(X[0])

=========================================

# Initialize the variable
Y = transform_text_to_sequences(en_sentences, output_tokenizer)

# Temporary list
ylist = list()
for sequence in Y:
  	# One-hot encode sentence and append to list
    ylist.append(to_categorical(sequence, num_classes=en_vocab_size))

# Update the variable
Y = np.array(ylist).reshape(Y.shape[0], Y.shape[1], en_vocab_size)

# Print the raw sentence and its transformed version
print("Raw sentence: {0}\nTransformed: {1}".format(en_sentences[0], Y[0]))

========================================

# Function to predict many phrases
def predict_many(model, sentences, index_to_word, raw_dataset):
    for i, sentence in enumerate(sentences):
        # Translate the Portuguese sentence
        translation = predict_one(model, sentence, index_to_word)
        
        # Get the raw Portuguese and English sentences
        raw_target, raw_src = raw_dataset[i]
        
        # Print the correct Portuguese and English sentences and the predicted
        print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))

predict_many(model, X_test[:10], en_index_to_word, test)

========================================











































