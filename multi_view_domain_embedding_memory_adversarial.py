from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys
import time
import numpy 
import tensorflow as tf
import read_data
from random import shuffle
import random
import numpy as np

class EmbeddingModel(object):

    def __init__(self, is_training, config, session):
        batch_size = config.batch_size
        num_steps = config.num_steps
        hidden_size= config.hidden_size
        vocab_size = config.vocab_size

        #inputs: features, mask and labels
        self.input_data = tf.placeholder(tf.int32, [num_steps, batch_size], name="inputs")
        self.mask= tf.placeholder(tf.int64, [batch_size], name="mask")
        self.labels=tf.placeholder(tf.int64, [batch_size], name="labels")
        self.domains=tf.placeholder(tf.int64, [batch_size], name="domains")
        self.memory_location=tf.placeholder(tf.int64, [batch_size], name="memory_location")

        #word embedding layer
        with tf.device("/cpu:0"):
            self.embedding=embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
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
        #num_step * batch_size * (hidden_size, hidden_size)
        self.lstm_output=tf.concat(lstm_output[0], 2)

class Domain_classifier:
    def __init__(self, share_model, weight1, bias1, weight2, bias2, config, is_adversarial=False):
        self.batch_size = config.batch_size
        self.share_model=share_model
        representation=tf.reduce_mean(share_model.lstm_output, axis=0)
        representation=tf.nn.relu(tf.matmul(representation, weight1) + bias1)
        logits=tf.matmul(representation, weight2) + bias2
        self.logits=logits


        #operators for prediction
        self.prediction=prediction=tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, share_model.domains)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))     

        #loss function
        global domain_size
        if is_adversarial:
        	loss=tf.nn.softmax(logits)*tf.one_hot(share_model.domains, depth=domain_size, on_value=0.0, off_value=1.0)
        else:
        	loss=tf.nn.softmax(logits)*tf.one_hot(share_model.domains, depth=domain_size, on_value=1.0, off_value=0.0)
        
        loss=tf.reduce_sum(loss,axis=1)
        loss=-tf.log(loss+1e-30)
        self.cost=cost =tf.reduce_sum(loss)

        #designate training variables
        tvars=tf.trainable_variables()
        if not is_adversarial:
            train_vars = [var for var in tvars if 'domain_classifier' in var.name]
            print("domain_classifier")
        else:
            train_vars = [var for var in tvars if 'shared_model/embedding' in var.name or 'bidirectional_rnn' in var.name]
            print("adversarial_domain_classifier")

        for tv in train_vars:
            print(tv.name)

        self.lr = tf.Variable(0.0, trainable=False)
        grads=tf.gradients(cost, train_vars)
        grads, _ = tf.clip_by_global_norm(grads,config.max_grad_norm)    
        optimizer = tf.train.AdagradOptimizer(self.lr)        
        self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

class Combine_two_model:
    def __init__(self, is_training,share_model, config, domain_embedding, num, memories, W_a, U_a, v_a,weight1, bias1, weight2, bias2, self_Q, self_K):
        self.share_model=share_model
        self.batch_size=batch_size=config.batch_size
        self.memory_location=memory_location= share_model.memory_location
        memory=memories[num]            
      
        #domain embedding layer     
        with tf.device("/cpu:0"):      
            #batch_size * (2*hidden_size)
            domain_inputs = tf.nn.embedding_lookup(domain_embedding, share_model.domains)
        
        #self attention
        self.score=tf.nn.softmax(tf.matmul(tf.matmul(domain_inputs, self_Q),tf.transpose(tf.matmul(domain_embedding, self_K))))
        self.domain_inputs= domain_inputs= tf.matmul(self.score, domain_embedding)

        #compute attention scores 
        #domain queries 
        query_vec=tf.matmul(domain_inputs, W_a)
        #replicate domain queries for num_steps and reshape
        query_vec=tf.reshape(tf.tile(tf.expand_dims(query_vec, dim=1), [1,config.num_steps,1]), [-1, 4*config.hidden_size])
        
        #reshape LSTM outputs to two-dimensional
        lstm_output=tf.transpose(share_model.lstm_output, [1, 0, 2])
        reshaped_lstm_output=tf.reshape(lstm_output, [-1, 2*config.hidden_size])

        #compute unnormalized scores
        layer1=tf.tanh(tf.add(query_vec, tf.matmul(reshaped_lstm_output, U_a)))
        unnormalized_scores=tf.reshape(tf.squeeze(tf.matmul(layer1, v_a),axis=[1]), [-1, config.num_steps])
        #in order to tackle variable length
        sequence_mask=tf.cast(tf.sequence_mask(share_model.mask, config.num_steps), tf.float32)
        minimize_softmax_score=sequence_mask*1e25-1e25
        unnormalized_scores=unnormalized_scores*sequence_mask+minimize_softmax_score
        #normalize the scores
        self.normalized_score=normalized_score=tf.nn.softmax(unnormalized_scores)

        #compute weighted vectors 
        normalized_score=tf.expand_dims(normalized_score, dim=2)
        combine_vector=tf.reduce_sum(normalized_score*lstm_output, axis=1)

        #update op for memory network
        self.update_memory=tf.scatter_update(memory, memory_location, combine_vector)

        #attention on memory samples
        self.samples=samples=tf.nn.softmax(tf.matmul(combine_vector,tf.transpose(memory)))
        self.context_vector= context_vector= tf.matmul(samples, memory)

        #concat both vectors
        combine_vector=tf.concat([context_vector, combine_vector],axis=1)

        #softmax matrix
        softmax_w = tf.get_variable("softmax_w", [4*config.hidden_size, 2])
        #softmax_w = tf.get_variable("softmax_w", [2*config.hidden_size, 2])
        softmax_b = tf.get_variable("softmax_b", [2])

        #add dropout to combine_vector
        if is_training and config.keep_prob < 1:
            combine_vector = tf.nn.dropout(combine_vector, config.keep_prob)

        logits = tf.matmul(combine_vector, softmax_w) + softmax_b

        #operators for prediction
        self.prediction=prediction=tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, share_model.labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        
        #cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=share_model.labels, logits=logits)
        cost = tf.reduce_sum(loss)

        self.cost=cost
        #compute grads and update 
        tvars=tf.trainable_variables()

        train_vars = [var for var in tvars if 'shared_model' in var.name or "m"+str(num) in var.name]

        print("m"+str(num))
        for tv in train_vars:
            print(tv.name)

        self.lr = tf.Variable(0.0, trainable=False)
        grads=tf.gradients(cost, train_vars)
        grads, _ = tf.clip_by_global_norm(grads,config.max_grad_norm)    
        optimizer = tf.train.AdagradOptimizer(self.lr)        
        self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    #assign value to learning rate
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

class Config(object):
    vocab_size=10000
    maxlen=100 
    num_steps = 100
    max_grad_norm = 5
    init_scale = 0.05
    hidden_size = 300
    lr_decay = 0.95
    valid_portion=0.1
    dataset='1'
    batch_size=50
    keep_prob = 0.4
    #0.05
    learning_rate = 0.1
    domain_learning_rate = 0.003
    max_epoch =2
    max_max_epoch =40

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

def run_pre_epoch(session, m, data, num):
    n_samples = data[0].shape[1]
    print("Running %d samples:"%(n_samples))  
    minibatches = get_minibatches_idx(n_samples, m.batch_size, shuffle=False)

    for inds in minibatches[:]:
        x = data[0][:,inds]        
        mask = data[1][inds]
        y = data[2][inds]
        memory_location= data[3][inds]

        memory_data=session.run([m.update_memory],\
            {m.share_model.input_data: x, m.share_model.mask: mask, m.share_model.labels: y,\
            m.share_model.domains: numpy.array([num]*len(y)), m.share_model.memory_location: memory_location})


def run_epoch(session, m, data, eval_op, num, is_training):
    n_samples = data[0].shape[1]
    print("Running %d samples:"%(n_samples))  
    minibatches = get_minibatches_idx(n_samples, m.batch_size, shuffle=False)

    correct = 0.
    total = 0
    total_cost=0
    for inds in minibatches[:]:
        x = data[0][:,inds]        
        mask = data[1][inds]
        y = data[2][inds]
        
        count, _, cost= \
        session.run([m.accuracy, eval_op,m.cost],\
            {m.share_model.input_data: x, m.share_model.mask: mask,m.share_model.labels: y,\
            m.share_model.domains: [num]*m.batch_size})

        correct += count 
        total += len(inds)
        total_cost+=cost

    print("Total loss:")
    print(total_cost)
    accuracy = correct/total
    return accuracy

def run_domain_classifier_epoch(session, m, data, eval_op):
    n_samples = data[0].shape[1]
    print("Running %d samples:"%(n_samples))  
    minibatches = get_minibatches_idx(n_samples, m.batch_size, shuffle=True)

    correct = 0.
    total = 0
    total_cost=0

    for inds in minibatches[:]:
        x = data[0][:,inds]        
        mask = data[1][inds]
        y = data[2][inds]

        count, _, prediction,cost, logits= \
        session.run([m.accuracy, eval_op, m.prediction, m.cost, m.logits],\
            {m.share_model.input_data: x, m.share_model.mask: mask, m.share_model.domains: y})

        correct += count 
        total += len(inds)
        total_cost+=cost

    print("Total loss:")
    print(total_cost)
    accuracy = correct/total
    return accuracy


def train_test_model(config, session, train_models, valid_models, test_models, trains, valids, tests, domain_classifier, domain_classifier_adversarial,combined_data):
    for i in range(config.max_max_epoch):
        #compute lr_decay
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        #zip the models and data
        model_list=list(zip(range(len(train_models)), train_models, valid_models, trains, valids))
        #reverse order 
        if i%2==1:
            model_list=reversed(model_list)

        #record which one has minimum training accuracies
        min_training=1.0
        number=-1
        for num, train_model, test_model, train, valid in model_list:
            #update memory            
            print("Updating Memories")
            run_pre_epoch(session, test_model, train, num=num)
            

            #update learning rate
            train_model.assign_lr(session, config.learning_rate * lr_decay)

            #training            
            print()
            print("Model: "+str(num+1))
            print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_model.lr)))
            start_time = time.time()
            train_acc = run_epoch(session, train_model, train, train_model.train_op, num=num, is_training=True)
            print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc, time.time()-start_time))
            
            #record mimimum training accuracy
            if train_acc< min_training:
                number=num
                min_training=train_acc
            
            #valid 
            valid_acc = run_epoch(session, test_model, valid, tf.no_op(), num=num, is_training=False)
            print("Valid Accuracy = %.4f\n" % valid_acc)

        #run model with minimum training accuracy again
        if number != -1:
            for num, train_model, test_model, train, valid in model_list:
                if num==number:
                    print("Model: "+str(num+1))
                    print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_model.lr)))
                    start_time = time.time()
                    train_acc = run_epoch(session, train_model, train, train_model.train_op, num=num, is_training=False)
                    print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc, time.time()-start_time))

            
        #testing
        for num, test_model, test in zip(range(len(test_models)),test_models, tests):          
            print(sys.argv[1+num])
            test_acc = run_epoch(session, test_model, test, tf.no_op(),num=num, is_training=False)
            print("Test Accuracy = %.4f\n" % test_acc)
            #write out accuracies
            with open("multi_view_domain.txt", 'a') as f:
                f.write("Accuracy for dataset "+ sys.argv[num+1]+": "+str(test_acc)+"\n")

        #domain classifier training
        print("Domain classifier Training:")
        domain_classifier.assign_lr(session, config.domain_learning_rate * lr_decay)
        start_time = time.time()
        domain_train_acc = run_domain_classifier_epoch(session, domain_classifier, combined_data, domain_classifier.train_op)
        print("Domain Training Accuracy = %.4f, time = %.3f seconds\n"%(domain_train_acc, time.time()-start_time))


        print("Domain adversarial classifier Training:")
        domain_classifier_adversarial.assign_lr(session, config.domain_learning_rate * lr_decay)
        start_time = time.time()
        domain_train_acc = run_domain_classifier_epoch(session, domain_classifier_adversarial, combined_data, domain_classifier_adversarial.train_op)
        print("Domain Training Accuracy = %.4f, time = %.3f seconds\n"%(domain_train_acc, time.time()-start_time))

def word_to_vec(session,config, *args):
    f = open("vectors"+config.dataset, 'rb')
    matrix= numpy.array(pickle.load(f))
    print("word2vec shape: ", matrix.shape)
    for model in args:
        session.run(tf.assign(model.embedding, matrix))


#combine two datasets
def combine(dataset):
    flag=False
    for single_dataset in dataset:
        if flag==False:
            flag=True
            combined_data=[single_dataset[0], single_dataset[1],single_dataset[4]]
        else:
            combined_data=[numpy.concatenate([combined_data[0],single_dataset[0]], axis=1),numpy.concatenate([combined_data[1],single_dataset[1]]),\
            numpy.concatenate([combined_data[2],single_dataset[4]])]
    return combined_data

def get_domains():
    #domains to be processed
    domain_list=sys.argv[1:]
    domain_size=len(domain_list)
    print(domain_size)
    if domain_size<=0:
        print("No dataset")
        exit(1)
    return domain_size, domain_list

if __name__ == "__main__":
    #configs
    config = Config()
    domain_size, domain_list=get_domains()

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
        train = read_data.prepare_data(train_datasets[index][0], train_datasets[index][1], maxlen=config.maxlen, traindata=True, index=index)
        valid = read_data.prepare_data(valid_datasets[index][0], valid_datasets[index][1], maxlen=config.maxlen, traindata=False, index=index)
        test = read_data.prepare_data(test_datasets[index][0], test_datasets[index][1], maxlen=config.maxlen, traindata=False, index=index)
        train_datasets[index]=train
        valid_datasets[index]=valid
        test_datasets[index]=test
  
    combined_data=combine(train_datasets)   

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        initializer = tf.random_normal_initializer(0, 0.05)

        #attention weights
        with tf.variable_scope("shared_model"):
            #domain embedding
            domain_embedding = tf.Variable(tf.random_normal([domain_size, 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="domain_embedding")
            W_a = tf.Variable(tf.random_normal([2*config.hidden_size, 4*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="W_a")
            U_a = tf.Variable(tf.random_normal([2*config.hidden_size, 4*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="U_a")
            v_a = tf.Variable(tf.random_normal([4*config.hidden_size, 1], mean=0.0, stddev=0.1, dtype=tf.float32), name="v_a")

        
        #domain self-attention weights
        with tf.variable_scope("self_attention"):          
            self_Q = tf.Variable(tf.random_normal([2*config.hidden_size, 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="Q")
            self_K = tf.Variable(tf.random_normal([2*config.hidden_size, 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="K")


        #memory network
        memories=[]
        for index, train in enumerate(train_datasets):
            memory = tf.Variable(tf.random_normal([len(train[3]), 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),trainable=False ,name="memory"+str(index))
            memories.append(memory)

        #weights for domain classifier (adversarial training)
        with tf.variable_scope('domain_classifier'):
            domain_classifier_weight1 = tf.Variable(tf.random_normal([2*config.hidden_size, config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="domain_classifier1")
            domain_classifier_bias1 = tf.Variable(tf.random_normal([config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="domain_classifier_bias1")

            domain_classifier_weight2 = tf.Variable(tf.random_normal([config.hidden_size, domain_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="domain_classifier2")
            domain_classifier_bias2 = tf.Variable(tf.random_normal([domain_size], mean=0.0, stddev=0.1, dtype=tf.float32), name="domain_classifier_bias2")

        #print memory shape
        print("memory shape")
        for index,memory in enumerate(memories):
            print(sys.argv[1+index])
            print(memory.get_shape())
        
        #training model for shared weights
        with tf.variable_scope("shared_model", reuse=None, initializer=initializer):
            share_model_train = EmbeddingModel(True, config=config, session=session)
        #testing model for shared weights
        with tf.variable_scope("shared_model", reuse = True, initializer=initializer):
            share_model_test = EmbeddingModel(False, config=config, session=session)

        #domain classifier         
        domain_classifier=Domain_classifier(share_model_train, domain_classifier_weight1, domain_classifier_bias1,domain_classifier_weight2, domain_classifier_bias2,config, False)
        domain_classifier_adversarial=Domain_classifier(share_model_train, domain_classifier_weight1, domain_classifier_bias1,domain_classifier_weight2, domain_classifier_bias2,config, True)

        #build models
        train_models=[]
        test_models=[]
        for index in range(domain_size): 
            with tf.variable_scope("m"+str(index), reuse = None, initializer=initializer):
                train_model = Combine_two_model(True,share_model_train, config, domain_embedding, index, memories, W_a, U_a,v_a, domain_classifier_weight1,domain_classifier_bias1, domain_classifier_weight2,domain_classifier_bias2, self_Q, self_K)
            with tf.variable_scope("m"+str(index), reuse = True, initializer=initializer):
                test_model = Combine_two_model(False,share_model_test, config, domain_embedding, index, memories, W_a, U_a,v_a, domain_classifier_weight1,domain_classifier_bias1, domain_classifier_weight2,domain_classifier_bias2, self_Q, self_K)
            train_models.append(train_model)
            test_models.append(test_model)

        #print trainable variables
        for v in tf.trainable_variables():
            print(v.name)

        #initialize
        init = tf.global_variables_initializer()
        session.run(init)

        #initialize share model's embedding with word2vec
        word_to_vec(session,config, share_model_train)
        #train test model
        train_test_model(config, session,\
            train_models,test_models,test_models,\
            train_datasets,valid_datasets,test_datasets, domain_classifier,domain_classifier_adversarial,combined_data)

        final_domain_embedding=session.run(domain_embedding)
        np.save('domain_embedding.npy', final_domain_embedding) 