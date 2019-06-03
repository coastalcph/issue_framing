import _dynet as dn
import random
from models.vocabs import Word2Int
import numpy as np
from models import labels
from models.layers import SoftmaxLayer, EmbeddingLayer, RnnLayer, GradientReversalLayer
from models import utils
from models import evaluate
import logging
import os



"""
A multi-task model for classification of sequences
"""
class MTL_Adversarial_model(object):

    def __init__(self, num_layers, input_dim, hidden_dim, tasks, src_domain, main_task, adversarial_task, vocabularies, update_embs, exp_path, prediction_layer, additional_params):
        # parameter collection
        self.model = dn.Model()

        #self.basename = basename
        self.exp_path = exp_path

        # dimension of the word embeddings
        self.input_dim = input_dim

        # dimension of the rnn hidden states
        self.hidden_dim = hidden_dim

        # number of layers of the rnn
        self.num_layers = num_layers

        # the tasks
        self.tasks = tasks


        # the src domain
        self.src_domain = src_domain

        # the task-specific layer used to predict the main task in case the main task is not trainable
        if tasks[main_task].trainable:
            self.prediction_layer = main_task

        else:
            self.prediction_layer = prediction_layer

        # additional parameters, e.g. dynet parameters, that are not set for the model directly but should be stored when reporting results
        self.additional_params = additional_params

        # the name of the main task that is the target of optimization
        self.main_task = main_task

        self.vocabularies = vocabularies

        self.update_embs = update_embs

        # setup the shared rnn
        self.rnn = self.setup_rnn(model=self.model, num_layers=self.num_layers, input_dim=self.input_dim, hidden_dim=self.hidden_dim)

        # setup the embedding layers for each vocabulary, then associate each task with the respective embedding layer (some tasks (or all tasks) might share the same embedding layer)
        self.embedding_layers = {}
        for voc_name, vocab_builder in sorted(iter(self.vocabularies.items())):
            self.embedding_layers[voc_name] = self.setup_embedding_layer(model=self.model, emb_dim=self.input_dim, vocab_size=vocab_builder.vocab_size, layername='{}#emb'.format(voc_name), update_embs=self.update_embs, embs=vocab_builder.embeds)

        # associate each task with the respective embedding layer
        self.task2embedding_layers = {}
        for tid, task in sorted(iter(self.tasks.items())):
            self.task2embedding_layers[tid] = task.vocab_name


        # set up the task specific output layers for each task.
        # don't set up an output layer if there is no training data for the task
        self.output_layers = {}
        for tid, task in sorted(iter(self.tasks.items())):
            if task.trainable:
                self.output_layers[tid] = self.setup_output_layer(model=self.model, input_dim=self.hidden_dim, output_dim=task.num_classes, layername='{}#out'.format(task.task_name))

        # add an embedding layer for the adversarial data
        self.task2embedding_layers['adversarial'] = adversarial_task.vocab_name
        # add a special output layer for the adversarial
        # we model a binary output layer predicting domain
        self.output_layers['adversarial'] = self.setup_output_layer(model=self.model, input_dim=self.hidden_dim,
                                                                output_dim=2, layername='adversarialout')

        self.gradient_reversal_layer = self.setup_gradient_reversal_layer(model=self.model, input_dim=self.hidden_dim, output_dim=self.hidden_dim,
                                                                          layername='gr')


        #store all the model parameters in the model_params dict
        self._set_model_params()



    def _set_model_params(self):
        self.model_params = {}
        self.model_params['num_layers'] = self.num_layers
        self.model_params['input_dim'] = self.input_dim
        self.model_params['hidden_dim'] = self.hidden_dim
        self.model_params['tasks'] = [elm for elm in list(self.tasks.keys()) if elm != self.main_task]
        self.model_params['main_task'] = self.main_task
        self.model_params['exp_path'] = self.exp_path
        self.model_params['update_wembeds'] = self.update_embs

        # add the task parameters to the model parameter dict
        task_params = {}
        for tid, task in self.tasks.items():
            task_params[tid] = task.get_params()
        self.model_params['tasks'] = task_params

        # add the additional parameters that were passed to the constructor
        for params_name, params in self.additional_params.items():
            self.model_params[params_name] = params



    def setup_output_layer(self, model, input_dim, output_dim, layername):
        return SoftmaxLayer(model=model, input_dim=input_dim, output_dim=output_dim, layername=layername)

    def setup_embedding_layer(self, model, emb_dim, vocab_size, update_embs, embs, layername):
        return EmbeddingLayer(emb_dim=emb_dim, vocab_size=vocab_size, model=model, update_embs=update_embs, embs=embs, layername=layername)

    def setup_rnn(self, model, num_layers, input_dim, hidden_dim):
        return RnnLayer(model=model, num_layers=num_layers, input_dim=input_dim, hidden_dim=hidden_dim)

    def setup_gradient_reversal_layer(self, model, input_dim, output_dim, layername):
        return GradientReversalLayer(model=model, layername=layername)



    def __call__(self, task, input_seq):
        """
        calls the network on the input sequence
        embeds the element via the lookup matrix, passes it through the hidden states and computes the softmax on the output of the
        last hidden state
        """
        embedding_layer = self.embedding_layers[self.task2embedding_layers[task]]
        output_layer = self.get_output_layer(task)

        embedded_seq = embedding_layer(input_seq)
        rnn_output = self.rnn(embedded_seq)

        # apply gradient reversal layer
        if task == 'adversarial':
            rnn_output=self.gradient_reversal_layer(rnn_output)

        output = output_layer(rnn_output)
        return output

    def get_output_layer(self, task):
        """
        returns the output layer used to predict a task. If the task is trainable, it will be the task-specific output layer associated with that task
        if the task is not trainable, it will be predicted by the output layer associated with the task stored in the prediction_layer argument
        :param task:
        :return:
        """
        if task=='adversarial':
            return self.output_layers[task]
        if self.tasks[task].trainable:
            return self.output_layers[task]
        else:
            return self.output_layers[self.prediction_layer]


    def compute_loss_multilabel(self, task, seq, multi_y):
        """
        computes the loss for multi-label instances by summing over the negative log probabilities of all correct labels
        """
        out_probs = self(task, seq)
        losses = []
        for y in multi_y:
            assigned_prob = dn.pick(out_probs, y)
            losses.append(-dn.log(assigned_prob)/len(multi_y))
        return dn.esum(losses)


    def train_batched(self, tasks, batch_size, scale_gradient_factor, validation_data, seqs_trg, early_stopping, patience, num_epochs,
                           min_num_epochs, num_updates, prob_main_task, prob_adv):
        trainer = dn.SimpleSGDTrainer(self.model)

        # stores best observed validation accuracy
        val_best = 0
        # stores the number of iterations without improvement
        no_improvement = 0
        val_prev = 0

        for epoch in range(num_epochs):
            sum_losses = 0
            adversarial_loss = 0
            losses_prediction_task = []
            losses_aux_task = []
            batch_dict = self.generate_batches_across_tasks(tasks, batch_size)

            # number of updates is twice the length of the main task batch list
            num_updates = len(batch_dict[self.prediction_layer]) * 2
            print(num_updates)
            #logging.INFO('Number of updates to do: {}'.format(num_updates))
            # sample batches according to some schema
            update_counter = 0
            while update_counter <= num_updates:
                update_counter += 1

                # with prob 1-prob_adv, do a task update
                outcome = np.random.binomial(1, prob_adv, size=None)
                if outcome == 0:
                    task_id, batch_ids = self.sample_task_batch(batch_dict, prob_main_task=prob_main_task)
                    losses = []
                    dn.renew_cg()
                    # iterate through the batch
                    for i in batch_ids:
                        seq = tasks[task_id].train_seqs[i]
                        label = tasks[task_id].train_labels[i]
                        loss = self.compute_loss_multilabel(task_id, seq, label)
                        losses.append(loss)

                    batch_loss = dn.esum(losses) / len(batch_ids)
                    batch_loss_value = batch_loss.value()
                    batch_loss.backward()
                    trainer.update()
                    sum_losses += batch_loss_value

                    if task_id == self.prediction_layer:
                        losses_prediction_task.append(batch_loss_value)
                    else:
                        losses_aux_task.append(batch_loss_value)
                else:
                    # do adversarial step
                    losses = []
                    dn.renew_cg()
                    seqs, labels = self.generate_adversarial_batch(seqs_src=tasks[self.src_domain].train_seqs,
                                                                       seqs_trg=seqs_trg, batch_size=batch_size)
                    for i in range(len(seqs)):
                        seq = seqs[i]
                        label = labels[i]
                        loss = self.compute_loss_multilabel(task='adversarial', seq=seq, multi_y=label)
                        losses.append(loss)
                    batch_loss = dn.esum(losses) / len(seqs)
                    batch_loss_value = batch_loss.value()
                    batch_loss.backward()
                    trainer.update()
                    adversarial_loss += batch_loss_value

            # compute the validation accuracy to monitor early stopping
            # use the micro averaged f as criterion
            res = evaluate_model_predictions(self.predict(self.main_task, validation_data['seq']),
                                             validation_data['label'], validation_data['labelset'])
            f_avg = res['f_avg']
            logging.info(
                'Epoch {}. Sum loss: {}. Avg loss: {}. Avg loss predtask {}. Avg loss aux tasks: {}. No improv: {}. Best f_val: {}. Avg f_val: {}'.format(epoch, sum_losses, sum_losses/num_updates,
                                                                                                                                                 np.mean(losses_prediction_task), np.mean(losses_aux_task),
                                                                                                  no_improvement,
                                                                                                  val_best, f_avg))
            logging.info(
                'Epoch {}. Adv loss: {}. Avg loss: {}. Avg loss predtask {}. Avg loss aux tasks: {}. No improv: {}. Best f_val: {}. Avg f_val: {}'.format(
                    epoch, adversarial_loss, sum_losses / num_updates,
                    np.mean(losses_prediction_task), np.mean(losses_aux_task),
                    no_improvement,
                    val_best, f_avg))


            # init early stopping after min number of epochs
            if epoch == min_num_epochs - 1:
                val_prev = f_avg
                no_improvement = 0
                self.save(self.exp_path)

            # if early_stopping:
            if f_avg <= val_prev:
                no_improvement += 1
                if early_stopping:
                    if no_improvement >= patience and epoch > min_num_epochs:
                        break
            else:
                if epoch >= min_num_epochs:
                    self.save(self.exp_path)
                no_improvement = 0
                if f_avg >= val_best:
                    val_best = f_avg
                val_prev = f_avg

        return epoch, f_avg, sum_losses, no_improvement, val_best


    def predict(self, task, inputs):
        preds = []
        for input in inputs:
            dn.renew_cg()
            out = self(task, input)
            preds.append(np.argmax(out.npvalue()))
        return preds


    def save(self, path):
        # save the parameters
        abs_basename = os.path.join(path, 'model')
        utils.write_json(self.model_params, path + '/model_params.json')
        # save the embedding layers
        for layer in self.embedding_layers.values():
            layer.save(abs_basename, append=False)
        # save the output layers
        for layer in self.output_layers.values():
            layer.save(abs_basename)
        # save the shared rnn
        self.rnn.save(abs_basename)


    def load(self, path):

        logging.info('Loading the model parameters from disk at {}'.format(path))
        abs_basename = os.path.join(path,  'model')
        # load the embedding layers
        for layer in self.embedding_layers.values():
            layer.load(abs_basename)
        # load the output layers
        for layer in self.output_layers.values():
            layer.load(abs_basename)
        # load the shared rnn
        self.rnn.load(abs_basename)


    def generate_batches(self, num_seqs, batch_size):
        if batch_size > num_seqs:
            batch_size = num_seqs
        idxs = [i for i in range(num_seqs)]
        random.shuffle(idxs)
        batches = []
        batch = []
        l = 0
        while l < len(idxs):
            lb = 0
            while lb < batch_size and l < len(idxs):
                batch.append(idxs[l])
                lb += 1
                l += 1
            batches.append(batch)
            batch = []
        return batches

    def generate_batches_across_tasks(self, tasks, batch_size):
        """
        generates batches of all datasets for all tasks
        :param tasks:
        :param batch_size:
        :return:
        """
        batches = {}
        for tid, task in sorted(iter(tasks.items())):
            batches[tid] = self.generate_batches(len(task.train_seqs), batch_size)
        return batches


    def sample_task_batch(self, batch_dict, prob_main_task):
        main_task_id = self.prediction_layer
        other_task_ids = [elm for elm in sorted(iter(self.tasks.keys())) if elm != main_task_id and self.tasks[elm].trainable and elm != 'adversarial']

        # coinflip. either sample from the task defining the prediction layer or from the aux tasks
        outcome = np.random.binomial(1, prob_main_task, size=None)
        if outcome == 1 or len(other_task_ids) == 0:
            # draw from prediction layer task
            batch = batch_dict[main_task_id][random.sample([i for i in range(len(batch_dict[main_task_id]))], 1)[0]]

            return main_task_id, batch
        else:

            # draw from one of the other tasks
            task_id = other_task_ids[random.sample([i for i in range(len(other_task_ids))], 1)[0]]

            # print(batch_dict[task_id])
            batch = batch_dict[task_id][random.sample([i for i in range(len(batch_dict[task_id]))], 1)[0]]
            return task_id, batch


    def generate_adversarial_batch(self, seqs_src, seqs_trg, batch_size):
        # sample batch_size/2 instances from the source domain and batch_size/2 instances from the target domain
        labels = []
        seqs = []
        idxs_src = [i for i in range(len(seqs_src))]
        random.shuffle(idxs_src)
        idxs_trg = [i for i in range(len(seqs_trg))]
        random.shuffle(idxs_trg)
        num_adversarial_data = np.min([len(seqs_src)-1, len(seqs_trg)-1, int(np.round(batch_size/2.))])
        seqs += [seqs_src[i]  for i in idxs_src[:num_adversarial_data]]
        seqs += [seqs_trg[i]  for i in idxs_trg[:num_adversarial_data]]
        labels += [[0] for elm in idxs_src[:num_adversarial_data]]
        labels += [[1] for elm in idxs_trg[:num_adversarial_data]]

        # shuffle
        idxs = [i for i in range(len(labels))]
        random.shuffle(idxs)
        return [seqs[i] for i in idxs], [labels[i] for i in idxs]



def fit_sequences(data, wembed_file=None):
    """
    preprocesses the texts. generates a vocabulary from the sequences
    by using pretrained word embeddings or a generated vocabulary
    :param data: a data dict
    :param use_pretrained_wembeds: boolean indicating if pretrained embeddings should be used
    :param vocab_builder:
    :return:
    """
    # Generate the vocabularies
    if wembed_file is not None:
        vocab_builder = Word2Int(embeds_file=wembed_file)
    else:
        vocab_builder = Word2Int()
    vocab_builder.build_vocab(data)
    return vocab_builder


def transform_sequences(data, vocab_builder):
    """
    maps a word sequence to a sequence of indexes into an embedding matrix, either
    by using the fitted vocabulary
    :param data: a data dict
    :param use_pretrained_wembeds: boolean indicating if pretrained embeddings should be used
    :param vocab_builder:
    :return:
    """
    # Transform the texts to sequences of word ids
    seqs = []
    for seq in data:
        seqs.append(vocab_builder.transform_word2int(seq))
    return seqs




def prepare_labels(labeldata, labelset, binary=False, target_class=''):
    """
    maps the labels in the datadict to one hot vectors and categorical vectors, i.e. for class
    :param data:
    :param labelset:
    :param target_class:
    :return:
    """
    if binary:
        labels_hot = labels.map_to_one_hot_binary(labeldata, target_class)
        labels_cat = labels.map_to_cat_binary(labeldata, target_class)
    else:
        labels_hot = labels.map_to_one_hot_multilabel(labeldata, labelset)
        labels_cat = labels.map_to_cat_multilabel(labeldata, labelset)
    return labels_hot, labels_cat


def select(l, idx):
    return [l[i] for i in idx]





def evaluate_model_predictions(preds, gold, class_labels):
    """
    the predictions as categorical and the gold labels as one hot vectors
    :param preds:
    :param gold:
    :param class_labels:
    :return:
    """

    return evaluate.evaluate_multiclass(utils.pred_to_one_hot(preds, class_labels), utils.pred_to_one_hot(gold, class_labels), class_labels)




