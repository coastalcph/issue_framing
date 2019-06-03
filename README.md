# Issue Framing in Online Discussion Fora

This repository contains code and data for the experiments described in the paper 
Mareike Hartmann,Tallulah Jansen, Isabelle Augenstein and Anders Søgaard. 2019. Issue Framing in Online Discussion Fora. In Proceedings of NAACL.

# Requirements
Python3
numpy
DyNet 2.0.3

Pretrained word embeddings need to be put into the resource folder specified in the config. The embeddings used in the original experiments were 'glove.6B.100d' that can be downloaded here: https://nlp.stanford.edu/projects/glove/

# Usage


In order to run the multi-task model, run model/run_dialogue_experiments_mtl.py
e.g. python3 models/run_dialogue_experiments_mtl.py --setting development --config config/config.cfg --early_stopping --patience 5 --update_wembeds --update_rnn --main_task dialogue --concat smoking immigration samesex --prediction_layer immigration#samesex#smoking --eval_set dialogue

In order to run the adversarial model, run model/run_dialogue_experiments_mtl.py
e.g. python3 models/run_dialogue_experiments_mtl_adversarial.py --num_layers 2 --input_dim 100 --hidden_dim 100 --num_epochs 2 --batch_size 100 --setting development --config config/config.cfg --early_stopping --patience 5 --update_wembeds --update_rnn --main_task dialogue --concat smoking immigration samesex --prediction_layer immigration#samesex#smoking --eval_set dialogue



If you have questions or wish to obtain the Media Frames data please contact the authors at hartmann.cl.uni-heidelberg.de

Please cite the paper as

@inproceedings{hartmann2019
  author    = {Mareike Hartmann and
               Tallulah Jansen and
               Isabelle Augenstein and
               Anders S{\o}gaard},
  title     = {{Issue Framing in Online Discussion Fora}},
  booktitle   = {Proceedings of NAACL},
  pages    = {1401--1407},
  year      = {2019}
}


