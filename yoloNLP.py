from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D
import keras

#a mapping from integer indices to class labels
label_map = {'O':0,'Lead':1,'Position':2,'Evidence':3,'Claim':4,
'Concluding_Statement':5,'Counterclaim':6,'Rebuttal':7}

#----------------------
#CONSTRUCTING THE MODEL

#load the huffingface model and associated tokenizer
model_name = 'allenai/longformer-base-4096'
longformer_pretrained = TFAutoModelForTokenClassification.from_pretrained(model_name,num_labels=24,output_hidden_states=True) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TokenClassifier(keras.Model):
	'''
	This model takes tokenized text as input and outputs a 24 dimensional class probability vector for each token.
	The idea is to interpret this class probability vector as an embedding of the input tokens.
	'''
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.longformer = longformer_pretrained #load the pretrained model from huggingface

	def call(self,inputs):
		'''
		Runs input tokens through the longformer model and computes a softmax activation over classes.

		Parameters
		----------
		inputs : dictionary
			inputs['input_ids'] is a tensor with shape (batch_size,1024) containing token ids for each essay in the batch.
			inputs['attentions'] contains the corresponding attention masks
		
		Returns
		-------
		out : tensor	
			Contains predicted class probabilities (i.e. embeddings) for each token of the input text.
		
		'''
		longformer = self.longformer(inputs['input_ids'], 
		                             attention_mask=inputs['attentions'])
		out = tf.nn.softmax(longformer[0],axis=2) #add a softmax to convert from logits to class probabilities, thereby normalizing the inputs to the convolutional layer
		return out

class Inception(keras.Model):
	'''
	An Inception-like module for one-dimensional sequence inputs.  
	This layer performs four convolutions and outputs the concatenated result.
	The convlutions have window size 1, 3, 5, and 7.  The number of filters
	for each convolutional layer are taken as input to the constructor.
	'''
	def __init__(self,d1,d3,d5,d7,**kwargs):
		'''
		Constructs the Inception module

		Parameters
		----------
		d1, d3, d5, d7 : int
			Number of filters to use for each convolutional layer.
		'''
		super().__init__(**kwargs)
		self.conv1 = Conv1D(d1,1,strides=1,padding='same',activation='relu')
		self.conv3 = Conv1D(d3,3,strides=1,padding='same',activation='relu')
		self.conv5 = Conv1D(d5,5,strides=1,padding='same',activation='relu')
		self.conv7 = Conv1D(d7,7,strides=1,padding='same',activation='relu')

	def call(self,inputs):
		conv1 = self.conv1(inputs)
		conv3 = self.conv3(conv1)
		conv5 = self.conv5(conv1)
		conv7 = self.conv7(conv1)
		return tf.concat((conv1,conv3,conv5,conv7),axis=-1)

class Yolo1D(keras.Model):
	#A YOLO-like model for natural language processing.
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.inception1 = Inception(64,128,32,32)
		self.inception2 = Inception(192,208,48,64)
		self.inception3 = Inception(112,288,64,64)
		self.inception4 = Inception(256,320,128,128)
		self.inception5 = Inception(384,384,128,128)

		self.conv1 = Conv1D(18,4,strides=2,padding='same',activation = 'relu')
		self.conv2 = Conv1D(36,4,strides=2,padding='same',activation = 'relu')

		self.class_out = Dense(7,activation='softmax')
		self.start_out = Dense(4,activation='softmax')
		self.length_out = Dense(1,activation='relu')
		self.confidence_out = Dense(1,activation='sigmoid')
	
	def call(self,inputs): 
		#First run the inputs through a series of Inception layers to encode local structure
		incepted = self.inception5(self.inception4(self.inception3(self.inception2(self.inception1(inputs)))))
		
		#Next run through dilatory convoutional layers to reduce to the desired size (i.e. bin the tokens into cells)
		convolved = self.conv2(self.conv1(incepted))

		#Finally produce the regression outputs with a series of fully connected
		class_out = self.class_out(convolved)
		start_out = self.start_out(convolved)
		length_out = self.length_out(convolved)
		confidence_out = self.confidence_out(convolved)
		
		#return the concatenated results
		return tf.concat((class_out,start_out,length_out,confidence_out),axis=2)


class FullModel(keras.Model):
	#Stacks the token classifier (longformer) with the 1D YOLO model to create a full text processing pipeline.
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.token_classifier = TokenClassifier()
		self.yolo = Yolo1D()
	def call(self,inputs):
		predicted_proba = self.token_classifier(inputs)
		return self.yolo(predicted_proba)

def yolo_loss(y_true,y_pred):
	#Loss function for the YOLO regression.

	#weights for confidence and corrdinate loss
	lambda_coord = 5
	lambda_noobj = 0.5

	#extract the regression variables
	class_true, start_true, length_true, conf_true = tf.split(y_true,[7,4,1,1],-1)
	class_pred, start_pred, length_pred, conf_pred = tf.split(y_pred,[7,4,1,1],-1)

	conf_true = conf_true[:,:,-1]
	length_true = length_true[:,:,-1]
	conf_pred = conf_pred[:,:,-1]
	length_pred = length_pred[:,:,-1]

	#sum of classification errors (only over tokens that truly belong to a class)
	class_loss = tf.math.reduce_sum(tf.square(tf.boolean_mask(class_true,conf_true) - tf.boolean_mask(class_pred,conf_true)))
	
	#confidence loss in predicting whether or not a cell contains the beginning of a named entity
	#instances without a named entity are more common and are therefore give a reduced weight
	#so as not to dominate the training
	conf_loss = tf.math.reduce_sum(tf.square(tf.boolean_mask(conf_true,conf_true) - tf.boolean_mask(conf_pred,conf_true))) \
					+ lambda_noobj*tf.math.reduce_sum(tf.square(tf.boolean_mask(conf_true,1-conf_true) - tf.boolean_mask(conf_pred,1-conf_true)))
	
	#sum of squared error in predicting token start position
	start_loss = lambda_coord*tf.reduce_sum(tf.square(tf.boolean_mask(start_true,conf_true) - tf.boolean_mask(start_pred,conf_true)))
	
	#sum of squared error in predicting named entity loss
	length_loss = lambda_coord*tf.reduce_sum(tf.square(tf.sqrt(tf.boolean_mask(length_true,conf_true)) - tf.sqrt(tf.boolean_mask(length_pred,conf_true))))

	#return the sum of losses for the four regression tasks
	return conf_loss + class_loss + start_loss + length_loss

class F1_Metric(keras.metrics.Metric):
	'''
	This macro-f1 score enables us to meaningfully track training progress for a multiclass
	classification task with unbalanced labels.
	'''
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		#[tp,fp,fn]
		#Each counts array contains true positives, false positives, and false negatives (in that order) for one of the seven classes
		self.counts1 = self.add_weight('counts1',shape = [3], initializer='zeros',dtype=tf.float32)
		self.counts2 = self.add_weight('counts2',shape = [3],initializer='zeros',dtype=tf.float32)
		self.counts3 = self.add_weight('counts3',shape = [3],initializer='zeros',dtype=tf.float32)
		self.counts4 = self.add_weight('counts4',shape = [3],initializer='zeros',dtype=tf.float32)
		self.counts5 = self.add_weight('counts5',shape = [3],initializer='zeros',dtype=tf.float32)
		self.counts6 = self.add_weight('counts6',shape = [3],initializer='zeros',dtype=tf.float32)
		self.counts7 = self.add_weight('counts7',shape = [3],initializer='zeros',dtype=tf.float32)

	def update_state(self,y_true,y_pred,sample_weight=None):
		#extract the regression variables
		class_true, start_true, length_true, conf_true = tf.split(y_true,[7,4,1,1],-1)
		class_pred, start_pred, length_pred, conf_pred = tf.split(y_pred,[7,4,1,1],-1)

		conf_true = conf_true[:,:,-1]
		length_true = length_true[:,:,-1]
		conf_pred = conf_pred[:,:,-1]
		length_pred = length_pred[:,:,-1]

		def get_counts(i):
			#count true positives
			a = tf.logical_and(conf_pred>0.5,conf_true==1)
			b = tf.logical_and(tf.argmax(class_true,-1)==i,tf.argmax(class_pred,-1)==i)
			c = tf.divide(tf.minimum(length_true,length_pred),tf.maximum(length_true,length_pred))>=0.5
			ab = tf.logical_and(a,b)
			abc = tf.logical_and(ab,c)
			tp = tf.reduce_sum(tf.cast(abc,tf.float32))

			#count false positives
			a = tf.logical_and(tf.argmax(class_pred,-1)==i, conf_pred>0.5)
			b = tf.logical_or(tf.argmax(class_true,-1)!=i, conf_true==0)
			ab = tf.logical_and(a,b)
			fp = tf.reduce_sum(tf.cast(ab,tf.float32))

			#count false negatives
			a = tf.logical_or(tf.argmax(class_pred,-1)!=i, conf_pred<=0.5)
			b = tf.logical_and(tf.argmax(class_true,-1)==i, conf_true==1)
			ab = tf.logical_and(a,b)
			fn = tf.reduce_sum(tf.cast(ab,tf.float32))

			return tf.stack((tp,fp,fn))
	 
	 	#update the counts
	    self.counts1.assign_add(get_counts(1))
	    self.counts2.assign_add(get_counts(2))
	    self.counts3.assign_add(get_counts(3))
	    self.counts4.assign_add(get_counts(4))
	    self.counts5.assign_add(get_counts(5))
	    self.counts6.assign_add(get_counts(6))
	    self.counts7.assign_add(get_counts(7))
	    self.counts = tf.stack((self.counts1,self.counts2,self.counts3,self.counts4,self.counts5,self.counts6,self.counts7))

	def result(self):
		#compute the f-score for each class and return the average over classes
		tp = self.counts[:,0]
		fp = self.counts[:,1]
		fn = self.counts[:,2]
		f_score = tf.math.divide_no_nan(tp,tp + 0.5*(fp+fn))

		return tf.reduce_mean(f_score)

	def reset_state(self):
		#reset counts to zero
		tf.keras.backend.batch_set_value([(v, tf.zeros(3)) for v in self.variables])

#--------------------
#GENERATING A DATASET

#import the competition data
trainpath = '.../train.csv'
train = pd.read_csv(trainpath)

#preprocess the dataframe to contain lists instead of strings (will facilitate subsequent preprocessing steps)
def split_int(string):
	return [int(x) for x in string.split()]
def split_string(string):
	string = string.replace('Concluding Statement','Concluding_Statement')
	return [x for x in string.split()]

for x in ['starts','ends','landmarks']:
	train[x]=train[x].apply(split_int)
train.labels = train.labels.apply(split_string)

#Define a few functions to transform the data into useful model inputs
max_length=1024 #required length (in tokens) of the longformer input
def tokenize(x):
	'''
	Tokenize an input text.

	Parameters
	----------
	x : string
		The text to be tokenized.

	Returns
	-------
	ids : int array
		Token ids of the input text.
	offsets : int array
		The position of each token id, measured in number of words from beginning of the input text.
		Necessary for converting start positions of named entities from word-indexed (as given in the competition data)
		to token-indexed (as required by our model).
	attention : bool array
		Mask containing True where tokens exist and False where padded.
	'''
	tokens = tokenizer.encode_plus(x, return_tensors='tf', return_offsets_mapping=True, max_length=max_length, padding='max_length', truncation=True)
	ids, offsets, attention = tokens[0].ids, tokens[0].offsets, tokens[0].attention_mask
	return ids, offsets, attention

def generate_starts(x):
	#convert start positions of named entities from word-indexed to token-indexed
	offsets = x.offsets.numpy()
	starts = x.starts
	return tf.constant([np.argmin(abs(s-offsets)) for s in starts],dtype=tf.int32)

def generate_ends(x):
	#convert end positions of named entities from word-indexed to token-indexed
	offsets = x.offsets.numpy()
	ends = x.ends
	return tf.constant([np.argmin(abs(s-offsets)) for s in ends],dtype=tf.int32)

#Run the preprocessing functions on the train dataframe to produce model inputs
tmp = train.text.apply(tokenize)
train['tokens'] = tmp.apply(lambda x:tf.constant(x[0]))
train['offsets'] = tmp.apply(lambda x:tf.constant([a[0] for a in x[1]]))
train['attention_mask'] = tmp.apply(lambda x:tf.constant(x[2]))
train['token_starts'] = train.apply(generate_starts,axis=1)
train['token_ends'] = train.apply(generate_ends,axis=1)
train['token_lengths'] = train.token_ends-train.token_starts

#Generate true values for training
start_arr=np.zeros((len(train),256))
class_arr=np.ones((len(train),256))
length_arr = np.zeros((len(train),256)) 
confidence_arr=np.zeros((len(train),256)) 
for i in range(len(train)):
	for j,lab in enumerate(train.labels.iloc[i]):
		ind = int(np.floor(train.token_starts.iloc[i][j]/4))
		start_val = np.mod(train.token_starts.iloc[i][j],4)
		class_arr[i,ind] = label_map[lab]
		start_arr[i,ind] = start_val
		length_arr[i,ind] = train.token_lengths.iloc[i][j]
		confidence_arr[i,ind] = 1

#load the data into a dataset
ds=tf.data.Dataset.from_tensor_slices((
	{'input_ids':tf.stack(train.tokens),
	'attentions':tf.stack(train.attention_mask)},
	tf.concat((
	tf.cast(tf.one_hot(tf.constant(class_arr,dtype=tf.int32),7),tf.float32),
	tf.cast(tf.one_hot(tf.constant(start_arr,dtype=tf.int32),4),tf.float32),
	tf.cast(tf.expand_dims(tf.constant(length_arr,dtype=tf.float32),axis=2),tf.float32),
	tf.cast(tf.expand_dims(tf.constant(confidence_arr,dtype=tf.float32),axis=2),tf.float32)),
	axis=2)
	))

#split into a train (80%), test (10%), and cv (10%) set
size = len(train)
train_size = int(0.8*size)
test_size = int(0.1*size)
ds = ds.shuffle(1000,seed=12)
ds_train = ds.take(train_size)
ds_test = ds.skip(train_size).take(test_size)
ds_val = ds.skip(train_size + test_size)

#Prepare the data for training
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 1 #Can increase the batch size if running on a more powerful machine
ds_train = ds_train.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
ds_test = ds_test.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

#-----------------
#COMPILE AND TRAIN
macro_f1 = F1_Metric()
model = FullModel()
model.compile(loss=yolo_loss, optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),metrics=macro_f1)

#train the model
history = model.fit(ds_train, epochs=2, validation_data=ds_test)

