# Issue Framing in Online Discussion Fora

This repository contains code and data for the experiments described in the paper <br/>
Mareike Hartmann,Tallulah Jansen, Isabelle Augenstein and Anders Søgaard. 2019. Issue Framing in Online Discussion Fora. In Proceedings of NAACL.

# Requirements
Python3<br/>
numpy<br/>
DyNet 2.0.3<br/>

Pretrained word embeddings need to be put into the resource folder specified in the config. The embeddings used in the original experiments were 'glove.6B.100d' that can be downloaded here: https://nlp.stanford.edu/projects/glove/

# Usage


In order to run the multi-task model, run model/run_dialogue_experiments_mtl.py<br/>
e.g. python3 models/run_dialogue_experiments_mtl.py --setting development --config config/config.cfg --early_stopping --patience 5 --update_wembeds --update_rnn --main_task dialogue --concat smoking immigration samesex --prediction_layer immigration#samesex#smoking --eval_set dialogue<br/>

In order to run the adversarial model, run model/run_dialogue_experiments_mtl.py<br/>
e.g. python3 models/run_dialogue_experiments_mtl_adversarial.py --num_layers 2 --input_dim 100 --hidden_dim 100 --num_epochs 2 --batch_size 100 --setting development --config config/config.cfg --early_stopping --patience 5 --update_wembeds --update_rnn --main_task dialogue --concat smoking immigration samesex --prediction_layer immigration#samesex#smoking --eval_set dialogue<br/>



If you have questions or wish to obtain the Media Frames data please contact the authors at hartmann@di.ku.dk<br/>

Please cite the paper as<br/>

@inproceedings{hartmann2019<br/>
  author    = {Mareike Hartmann and<br/>
               Tallulah Jansen and<br/>
               Isabelle Augenstein and<br/>
               Anders S{\o}gaard},<br/>
  title     = {{Issue Framing in Online Discussion Fora}},<br/>
  booktitle   = {Proceedings of NAACL},<br/>
  pages    = {1401--1407},<br/>
  year      = {2019}<br/>
}


