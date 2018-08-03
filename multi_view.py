from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys
import time
import numpy 
import tensorflow as tf
import read_data
from random import shuffle

class EmbeddingModel(object):

    def __init__(self, is_training, config, session, trainable):
        batch_size = config.batch_size
        #the steps of applying LSTM
        num_steps = config.num_steps
        hidden_size= config.hidden_size
        vocab_size = config.vocab_size

        #inputs: features, mask and labels
        self.input_data = tf.placeholder(tf.int32, [num_steps, batch_size], name="inputs")
        self.mask= tf.placeholder(tf.int64, [batch_size], name="mask")
        self.labels=tf.placeholder(tf.int64, [batch_size], name="labels")
        self.domains=tf.placeholder(tf.int64, [batch_size], name="domains")

        #word embedding layer
        with tf.device("/cpu:0"):
            self.embedding=embedding = tf.get_variable("embedding", [vocab_size, hidden_size], trainable=trainable)
            # num_steps* batch_size * embedding_size
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            #add dropout to input units
            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        #add LSTM cell and dropout nodes
        with tf.variable_scope('forward'):
            fw_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
            if is_training and config.keep_prob < 1:
                fw_lstm = tf.contrib.rnn.DropoutWrapper(fw_lstm, output_keep_prob=config.keep_prob)

        with tf.variable_scope('backward'):
            bw_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
            if is_training and config.keep_prob < 1:
                bw_lstm = tf.contrib.rnn.DropoutWrapper(bw_lstm, output_keep_prob=config.keep_prob)

        #bidirectional rnn
        lstm_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=inputs, sequence_length=self.mask, time_major=True, dtype=tf.float32)
        #num_step * batch_size * (hidden_size, hidden_siz)
        self.lstm_output=lstm_output=tf.concat(lstm_output[0], 2)
        #final sentence embedding.  batch_size * (2 * hidden_size)
        self.lstm_output=lstm_output=tf.reduce_mean(lstm_output, axis=0)

class Combine_two_model:
    def __init__(self, share_model, config):
        self.share_model=share_model
        self.batch_size=batch_size=config.batch_size
        
        #combined_embedding=tf.concat([model.lstm_output, share_model.lstm_output],1)
        #softmax matrix
        softmax_w = tf.get_variable("softmax_w", [2*config.hidden_size, config.num_classes])
        softmax_b = tf.get_variable("softmax_b", [config.num_classes])
        logits = tf.matmul(share_model.lstm_output, softmax_w) + softmax_b
        #cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=share_model.labels, logits=logits)
        self.entropy=cost = tf.reduce_sum(loss)
        #add regularization
        tvars = tf.trainable_variables()
        for var in tvars:
            if ('shared_model/bidirectional_rnn' in var.name and 'biases' not in var.name) \
            or 'shared_model/embedding' in var.name or tf.get_variable_scope().name+'/embedding' in var.name:
                cost=tf.add(cost, get_lambda(var.name, config)*tf.nn.l2_loss(var))
        self.cost= cost
        #operators for prediction
        self.prediction=prediction=tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, share_model.labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        
        #operators for optimizer
        self.lr = tf.Variable(0.0, trainable=False)
        
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        self.grads=grads[4]
        optimizer = tf.train.AdagradOptimizer(self.lr)
        #optimizer = tf.train.AdamOptimizer(self.lr)
        #self.train_op = optimizer.minimize(cost)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    #assign value to learning rate
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

class Config(object):
    vocab_size=10000  # Vocabulary size
    maxlen=100  # Sequence longer then this get ignored
    num_steps = maxlen
    batch_size=10  # The batch size during training.

    init_scale = 0.05
    learning_rate = 1
    max_grad_norm = 5
    hidden_size = 300
    max_epoch = 1
    max_max_epoch =30
    keep_prob = 0.40
    lr_decay = 0.90
    lambda_loss_m1=3e-6
    lambda_loss_m2=3e-6
    lambda_loss_share=3e-6
    valid_portion=0.1
    domain_size=2
    dataset='1'

#get lambda for regularization
def get_lambda(name, config):
	if "m1" in name:
		return config.lambda_loss_m1
	if "m2" in name:
		return config.lambda_loss_m2
	if "shared_model" in name:
		return config.lambda_loss_share
def get_minibatches_idx(n, batch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[-batch_size:])
    return minibatches


def run_epoch(session, m, data, eval_op, num=1000):
    n_samples = data[0].shape[1]
    print("Running %d samples:"%(n_samples))  
    minibatches = get_minibatches_idx(n_samples, m.batch_size, shuffle=False)

    correct = 0.
    total = 0
    #predictions
    p=[]
    total_entropy=0
    total_cost=0
    for inds in minibatches[:]:
        x = data[0][:,inds]        
        mask = data[1][inds]
        y = data[2][inds]
        
        count, _, prediction,embedding, cost, entropy, grads= \
        session.run([m.accuracy, eval_op, m.prediction, m.share_model.embedding, m.cost, m.entropy, m.grads],\
            {m.share_model.input_data: x, m.share_model.mask: mask, m.share_model.labels: y,\
            m.share_model.domains: numpy.array([num]*len(y))})
        print(grads)
        correct += count
        total += len(inds)
        p+=prediction.tolist()
        total_entropy+=entropy
        total_cost+=cost

    print("Entropy loss")
    print(total_entropy)
    print("Total loss:")
    print(total_cost)
    accuracy = correct/total
    return (accuracy, p)

def load_dataset(path, config):
    print('Loading data: '+ path)
    train, valid, test = read_data.load_data(path, n_words=config.vocab_size, \
        valid_portion=0.15, maxlen=config.maxlen)
    train = read_data.prepare_data(train[0], train[1], maxlen=config.maxlen)
    valid = read_data.prepare_data(valid[0], valid[1], maxlen=config.maxlen)
    test = read_data.prepare_data(test[0], test[1], maxlen=config.maxlen)
    return (train, valid, test)

def train_test_model(config, session, train_models, valid_models, test_models, trains, valids, tests):
    for i in range(config.max_max_epoch):
        #compute lr_decay
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
       	model_list=list(zip(range(len(train_models)), train_models, valid_models, trains, valids))
       	if i%2==0:
            model_list=reversed(model_list)
        min_training=1.0
        number=-1
        for num, train_model, test_model, train, valid in model_list:
            #update learning rate
            train_model.assign_lr(session, config.learning_rate * lr_decay)
            print("")
            print("Model: "+str(num+1))
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model.lr)))
            start_time = time.time()
            if(train_model.__class__.__name__=='Combine_two_model'):
                train_acc = run_epoch(session, train_model, train, train_model.train_op, num=num)
            print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc[0], time.time()-start_time))
            
            if train_acc[0] < 0.9 and train_acc[0]< min_training:
                number=num
                min_training=train_acc[0]

            
            valid_acc = run_epoch(session, test_model, valid, tf.no_op(), num=num)
            print("Valid Accuracy = %.4f\n" % valid_acc[0])

        if number != -1:
            for num, train_model, test_model, train, valid in model_list:
                if num==number:
                    print("Model: "+str(num+1))
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model.lr)))
                    start_time = time.time()
                    train_acc = run_epoch(session, train_model, train, train_model.train_op, num=num)
                    print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc[0], time.time()-start_time))

            
            #print(valid_acc[1])
        for num, test_model, test in zip(range(len(test_models)),test_models, tests):          
            test_acc = run_epoch(session, test_model, test, tf.no_op(),num=num)
            
            print(sys.argv[1+num])
            print("Test Accuracy = %.4f\n" % test_acc[0])
            
            with open("multi_result_final.txt", 'a') as f:
                f.write("final accuracy for dataset "+ sys.argv[num+1]+": "+str(test_acc[0])+"\n")


#combine two datasets
def combine(d1, d2):
    return numpy.concatenate([d1[0],d2[0]], axis=1),\
    numpy.concatenate([d1[1],d2[1]]),numpy.concatenate([d1[2],d2[2]])

def word_to_vec(session,config, *args):
    f = open("vectors"+config.dataset, 'rb')
    #f = open("domainvectors", 'rb')
    matrix= numpy.array(pickle.load(f))
    print("word2vec shape: ", matrix.shape)
    for model in args:
        session.run(tf.assign(model.embedding, matrix))

def extend(train, times):
    newtrain=train
    for i in range(times-1):
        newtrain=combine(newtrain, train)
    return newtrain

#make dataset approximately the same size
def extend_data(train, train1):
    if train[0].shape[0] > train1[0].shape[0]:
        if train[0].shape[0]/train1[0].shape[0]>1:
            train1=extend(train1, train[0].shape[0]/train1[0].shape[0])
        elif float(train[0].shape[0])/train1[0].shape[0]>1.6:
            train1=extend(train1, 2)
    else:
        if train1[0].shape[0]/train[0].shape[0]>1:
            train=extend(train, train1[0].shape[0]/train[0].shape[0])
        elif float(train1[0].shape[0])/train[0].shape[0]>1.6:
            train=extend(train, 2)
    return train, train1

def count_labels(labels):
    return len(set(labels))

def main(unused_args):
    #configs
    config = Config()
    #domains to be processed
    domain_list=sys.argv[1:]
    domain_size=len(domain_list)
    if domain_size<=0:
        print("No dataset")
        exit(1)
    #load dataset
    train_datasets, valid_datasets, test_datasets=[],[],[]
    for domain in domain_list:
        train, valid, test = read_data.load_data(path='dataset'+config.dataset+'/'+domain+'/dataset',n_words=config.vocab_size, \
            valid_portion=config.valid_portion, maxlen=config.maxlen)
        train_datasets.append(train)
        valid_datasets.append(valid)
        test_datasets.append(test)
    #transform dataset to matrix
    for index in range(domain_size):
        train = read_data.prepare_data(train_datasets[index][0], train_datasets[index][1], maxlen=config.maxlen, traindata=True)
        valid = read_data.prepare_data(valid_datasets[index][0], valid_datasets[index][1], maxlen=config.maxlen, traindata=False)
        test = read_data.prepare_data(test_datasets[index][0], test_datasets[index][1], maxlen=config.maxlen, traindata=False)
        train_datasets[index]=train
        valid_datasets[index]=valid
        test_datasets[index]=test

    config.num_classes = count_labels(train_datasets[0][2])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        initializer = tf.random_normal_initializer(0, 0.05)

        #training model for shared weights
        with tf.variable_scope("shared_model", reuse=None, initializer=initializer):
            share_model_train = EmbeddingModel(is_training=True, config=config, session=session,trainable=True)
        #testing model for shared weights
        with tf.variable_scope("shared_model", reuse = True, initializer=initializer):
            share_model_test = EmbeddingModel(is_training=False, config=config, session=session, trainable=True)

        #build models
        train_models=[]
        test_models=[]
        for index in range(domain_size): 
            with tf.variable_scope("m"+str(index), reuse = None, initializer=initializer):
                train_model = Combine_two_model(share_model_train, config)
            with tf.variable_scope("m"+str(index), reuse = True, initializer=initializer):
                test_model = Combine_two_model(share_model_test, config)
            train_models.append(train_model)
            test_models.append(test_model)

        init = tf.global_variables_initializer()
        session.run(init)

        #initialize share model's embedding with word2vec
        word_to_vec(session,config, share_model_train)
        #train test model
        train_test_model(config, session,\
            train_models,test_models,test_models,\
            train_datasets,valid_datasets,test_datasets)

if __name__ == '__main__':
    tf.app.run()
