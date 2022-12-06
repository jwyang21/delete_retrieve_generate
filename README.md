# Description

This is an implementation of the DeleteOnly and DeleteAndRetrieve models from [Delete, Retrieve, Generate:
A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf)

# Flow
installation -> preprocess -> (optional) check-process-data -> data prep -> train -> inference


# Installation     

`pip3 install -r requirements.txt`    

## My implementation
위에 나와있는 설치 command대로 했을 때, 아래 사진처럼 에러가 발생함.
            
![install-error](https://user-images.githubusercontent.com/86412887/205253321-45e84c16-04ac-43d6-9b89-717cecd8ae10.png)        

'pip install tensorboardX' 로 패키지 개별 설치.        
이 외에도 `pip3 install -r requirements.txt` 썼을 때 제대로 설치되지 않는 패키지들에 대해서는, 아래 command로 **mamba** 설치 후 mamba install ~~ 로 개별 패키지 설치해서 사용     
- conda install mamba -n [환경이름] -c conda-forge      


추가적으로, langdetect 패키지 사용 (https://pypi.org/project/langdetect/)
- pip install langdetect          


torch 설치의 경우, 아래 command 사용 (cpu서버)           
- mamba install pytorch torchvision torchaudio cpuonly -c pytorch


conda 환경 (cpu) yml 파일
- tst.yml

# Preprocess
baseline 코드에는 없고, poem 및 reddit 데이터 쓰기 위해 추가함.       
- python3 preprocess.py    

(optional) check preprocesss <- small trainset으로 모델 돌리는 경우에만!        
small trainset 기준으로, poem 데이터의 small trainset에 있는 모든 문장이 poem 전체 corpus에 속하는지 확인. 같은 작업을 reddit에 대해서도 함.               
- python3 check-processed-data.py            



# Data prep
Given two pre-tokenized corpus files, use the scripts in `tools/` to generate a vocabulary and attribute vocabulary:        

```
python tools/make_vocab.py [entire corpus file (src + tgt cat'd)] [vocab size] > vocab.txt
python tools/make_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
python tools/make_ngram_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
```

## My implementation       
Make vocabulary set of entire train corpus (num_vocab: 10000)          
- python tools/make_vocab.py data/processed/entire_small_train_corpus.txt 10000 > data/processed/entire_small_train_vocab.txt         

- python tools/make_vocab.py data/processed/small_poem_train_corpus.txt 10000 > data/processed/poem_small_train_vocab.txt                 

- python tools/make_vocab.py data/processed/small_reddit_train_corpus.txt 10000 > data/processed/reddit_small_train_vocab.txt                 

Find attribute markers (ngrams) from poem and reddit train corpus, respectively (saliency ratio: 5.5)               
- poem attribute ngram: python tools/make_ngram_attribute_vocab.py data/processed/entire_small_train_vocab.txt data/processed/small_poem_train_corpus.txt data/processed/small_reddit_train_corpus.txt 5.5 > data/processed/small_poem_attribute_ngram_vocab_s5.5.txt                   
- reddit attribute ngrams: python tools/make_ngram_attribute_vocab.py data/processed/entire_small_train_vocab.txt data/processed/small_reddit_train_corpus.txt data/processed/small_poem_train_corpus.txt 5.5 > data/processed/small_reddit_attribute_ngram_vocab_s5.5.txt  


# Training (runs inference on the dev set after each epoch)

`python3 train.py --config yelp_config.json --bleu`

This will reproduce the _delete_ model on a dataset of yelp reviews:

![curves](https://i.imgur.com/jfYaDBr.png)

Checkpoints, logs, model outputs, and TensorBoard summaries are written in the config's `working_dir`.

See `yelp_config.json` for all of the training options. The most important parameter is `model_type`, which can be set to `delete`, `delete_retrieve`, or `seq2seq` (which is a standard translation-style model).

## My implementation

전처리된 데이터 써서 train
- poem2reddit: python3 train.py --config p2r_e15.json --bleu

# Inference

`python inference.py --config yelp_config.json --checkpoint path/to/model.ckpt`

To run inference, you can point the `src_test` and `tgt_test` fields in your config to new data.

## My implementation
poem2reddit: python inference.py --config p2r_e15.json --checkpoint p2r_e15/model.1.ckpt

# etc
결과파일들이 총 4종류 (auxs.{i}, golds{i}, inputs.{i}, predicts.{i})                        
          
지금까지 파악하기로는 (확실치는 않지만) 아래와 같은 것으로 보임                   
- inputs.{i}: i-th epoch 때 모델 디코더 (text style transfer 실행하는 부분)에 input된 문장                    
- preds.{i}: i-th epoch 때 모델 디코더가 output한 최종 문장        
- golds.{i}: desired output               


# Citation

If you use this code as part of your own research can you please cite 

(1) the original paper:
```
@inproceedings{li2018transfer,
 author = {Juncen Li and Robin Jia and He He and Percy Liang},
 booktitle = {North American Association for Computational Linguistics (NAACL)},
 title = {Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer},
 url = {https://nlp.stanford.edu/pubs/li2018transfer.pdf},
 year = {2018}
}

```

(2) The paper that this implementation was developed for:
```
@inproceedings{pryzant2020bias,
 author = {Pryzant, Reid and Richard, Diehl Martinez and Dass, Nathan and Kurohashi, Sadao and Jurafsky, Dan and Yang, Diyi},
 booktitle = {Association for the Advancement of Artificial Intelligence (AAAI)},
 link = {https://nlp.stanford.edu/pubs/pryzant2020bias.pdf},
 title = {Automatically Neutralizing Subjective Bias in Text},
 url = {https://nlp.stanford.edu/pubs/pryzant2020bias.pdf},
 year = {2020}
}
```


# FAQ

### Why can't I get the same BLEU score as the original paper? 

- My script just runs in one direction (e.g. pos => neg). Maybe running the model in both directions (pos => neg, neg => pos) and then averaging the BLEU would get closer to their results
- The [implementation of BLEU that the original paper used](https://github.com/lijuncen/Sentiment-and-Style-Transfer/blob/250d22d39607bf697082861af935ab8e66e2160c/src/test_tool/BLEU/my_bleu_evaluate.py) has bugs in it and does not report correct BLEU scores. For example, it disagrees with [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) which is a canonical implementation of BLEU. If you use their script on our outputs you get something more similar (I think ~7.6 ish) but again their script might not be producing correct BLEU scores. 

### Why does `delete_retrieve` run so slowly? 

- The system runs a similarity search over the entire dataset on each training step. Precomputing some of these similarities would definitely speed things up if people are interested in contributing!

### What does the salience ratio mean? How was your number chosen?

- Intuitively the salience ratio says how strongly associated with each class do you want the attribute ngrams to be. Higher numbers means that the attribute vocab will be more strongly associated with each class, but also that you will have fewer vocab items because the threshold is tighter.
- The example attributes in this repo use the ratios from the paper, which were selected manually using a dev set. 


### I keep getting `IndexError: list index out of range` errors! 

- There is a known bug where the size of the A and B datasets need to match each other (again a great place to contribute!). Since the corpora don't need to be in alignment you can just duplicate some examples or trim one of the datasets so that they match each other. 

### I keep getting `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` errors!
The pytorch version this has been tested on is 1.1.0, which is compatible with cudatoolkit=9.0/10.0. If your cuda version is newer than this you may get the above error. Possible fix: 
```
$ conda install pytorch==1.1.0 torchvison==0.3.0 cudatoolkit=10.0 -c pytorch
```


# Acknowledgements

Thanks lots to [Karishma Mandyam](https://github.com/kmandyam) for contributing! 
