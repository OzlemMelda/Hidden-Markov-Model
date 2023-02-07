# HMM
## Sequential Text Classification
Classifying IMDB reviews as positive and negative using HMMs

*folder structure
- Project3
   - data
   - models (trained and saved models (maxiter_hiddenstates_rowcount_vocabsize))
      - 30_1_2500_256
      - 30_5_2500_256
      - 30_10_2500_256
      - 30_20_2500_256
   - scripts
      - HMM.py
      - classify.py

*HMM.py: training HMM

- parameters:
-- dev_path: Path to development (i.e., training, testing) data
-- train_path: Path to the training data directory
-- max_iters: The maximum number of EM iterations (default 30)
-- model_out: Filename to save the final model
-- dict_out: File name to save dictionary
-- hidden_states: The number of hidden states to use. (default 10)
-- row_count: The number of training data to use. (default 2500)
-- vocab_size: The number words in the frequent word dictionary, others are Unknown Words. (default 256)

- command samples:
--Train for positive HMM Hidden states: 1, Row count: 2500, Max iteration: 30, Vocab size: 256:
python ./scripts/HMM.py --dev_path ./data --train_path ./train/pos --max_iters 30 --model_out ./models/positive_1_2500_30_256.pkl --dict_out ./models/dictionary_positive_1_2500_30_256.pkl --hidden_states 1 --row_count 2500 --vocab_size 256

--Train for negative HMM Hidden states: 1, Row count: 2500, Max iteration: 30, Vocab size: 256:
python ./scripts/HMM.py --dev_path ./data --train_path ./train/neg --max_iters 30 --model_out ./models/negative_1_2500_30_256.pkl --dict_out ./models/dictionary_negative_1_2500_30_256.pkl --hidden_states 1 --row_count 2500 --vocab_size 256

--Train for positive HMM Hidden states: 5, Row count: 1000, Max iteration: 30, Vocab size: 256:
python ./scripts/HMM.py --dev_path ./data --train_path ./train/pos --max_iters 30 --model_out ./models/positive_5_1000_30_256.pkl --dict_out ./models/dictionary_positive_5_1000_30_256.pkl --hidden_states 5 --row_count 1000 --vocab_size 256

--Train for negative HMM Hidden states: 5, Row count: 1000, Max iteration: 30, Vocab size: 256:
python ./scripts/HMM.py --dev_path ./data --train_path ./train/neg --max_iters 30 --model_out ./models/negative_5_1000_30_256.pkl --dict_out ./models/dictionary_negative_5_1000_30_256.pkl --hidden_states 5 --row_count 1000 --vocab_size 256

*classify.py: testing HMM / classifying by HMM

- parameters:
-- pos_hmm: Path to the positive class hmm
-- neg_hmm: Path to the negative class hmm
-- datapath: Path to the test data
-- pos_dictpath: Path to the positive hmm dictionary
-- neg_dictpath: Path to the negative hmm dictionary

- command samples:
--Classify for Hidden states: 1, Row count: 2500, Max iteration: 30, Vocab size: 256:
python ./scripts/classify.py --pos_hmm ./models/positive_1_2500_30_256.pkl --neg_hmm ./models/negative_1_2500_30_256.pkl --datapath ./data/test --pos_dictpath ./models/dictionary_positive_1_2500_30_256.pkl --neg_dictpath ./models/dictionary_negative_1_2500_30_256.pkl

--Classify for Hidden states: 5, Row count: 500, Max iteration: 30, Vocab size: 256:
python ./scripts/classify.py --pos_hmm ./models/positive_5_1000_30_256.pkl --neg_hmm ./models/negative_5_1000_30_256.pkl --datapath ./data/test --pos_dictpath ./models/dictionary_positive_5_1000_30_256.pkl --neg_dictpath ./models/dictionary_negative_5_1000_30_256.pkl

*Important Notes:

- Libraries related
-- I used nltk library just to clean English words from the reviews (I, you, she, he, ...) NOT use it in my model.

- Dictionary related
-- Dictionary is created in HMM.py to represent 256 frequent words with an index. (1 is for UNK)
-- Two different dictionaries are created for two different HMM models; one for positive HMM, other one for negative HMM.
-- I put these two dictionaries in submitted folder. HMM.py automatically creates these files, however if you want to test my trained models using classify.py, you will need these dictionaries. That's why I put these files.

- Memory related
-- Defined a parameter 'row_count'. This parameter defines the amount of training to use. I have trouble in training model using all data in terms of memory and time. That's why I need this parameter.

- Folder structure related
-- I put the models that I trained into the subfolders under models folder. 
-- The models you will train will be directly saved to models folder. (no subfolder)

*Running recommendations

-- If you want to test my HMM.py file to see it running, set row_count to a smaller number (100-500) OR set row_count to 2500 and hidden states to small number (1-3), since it takes too much time to train the model using all data with high number of hidden states.
-- Trained my models using 2500 positive and 2500 negative reviews due to memory and time cost of my PC. It took too much time, too.
