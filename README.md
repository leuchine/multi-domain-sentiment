multi-domain-sentiment
======
A framework for multi-domain sentiment analysis by learning domain-specific representations of input sentences using neural network. 

Prerequisite
======
1. Tensorflow 
2. Google News Embeddings (https://code.google.com/archive/p/word2vec/) (rename it to 'vectors.gz')
3. Gensim

Data Preparation
======
1. Download datasets (e.g. laptops). We assume the datasets are preprocessed into the following format:

     The unit does everything it promises . I 've only used it once so far , but i 'm happy with it ||| 1

2. Randomly split each dataset into training (e.g. laptops/trn), development (e.g. laptops/dev) and testing datasets (e.g. laptops/tst). Put all datasets into a folder named 'dataset'. Thus, the directory structure looks like dataset/laptops/trn. 

3. Run `python preprocessing.py`. This program will iterate through the 'dataset' folder and generate files like dictionaries, embeddings and transformed datasets.

4. Run `python multi_view_domain_embedding_memory_adversarial.py dataset_name1 dataset_name2 ...` for running the algorithm.

 

