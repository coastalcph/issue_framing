import argparse
import _dynet as dn
from models import mtl_adversarial
from models import utils, constants
from models.vocabs import Word2Int
import configparser
import logging
import os
import random
import numpy as np


def setup_dynet(random_seed, weight_decay, mem):
    """
    sets the dynet parameters and returns a dictionary storing these parameters that can be passed to the model as additional parameters
    in order to store them
    :param random_seed:
    :param weight_decay:
    :param mem:
    :return:
    """
    dynet_params = {}
    dyparams = dn.DynetParams()

    dyparams.set_random_seed(random_seed)
    dynet_params['random_seed'] = random_seed

    dyparams.set_weight_decay(weight_decay)
    dynet_params['weight_decay'] = weight_decay

    dyparams.set_autobatch(True)
    dynet_params['autobatch'] = True

    dyparams.set_mem(mem)
    dynet_params['mem'] = mem

    # Initialize with the given parameters
    dyparams.init()
    return dynet_params




def load_data(config, main_task, aux_tasks, concat_datasets, trg_domain):
    """
    loads the data for all tasks specified via their task_names
    :param config:
    :param task_names:
    :return:
    """
    task2data = {}

    # load the data for the main task
    logging.info('Loading main task dataset: {}'.format(main_task))
    task2data[main_task] = _add_task_indicator(utils.load_json(config.get('Files', '{}_datasplit'.format(main_task))), main_task)

    # load the data for the auxiliary tasks
    if aux_tasks:
        for aux_task in aux_tasks:
            logging.info('Loading aux task dataset: {}'.format(aux_task))
            task2data[aux_task] = _add_task_indicator(utils.load_json(config.get('Files', '{}_datasplit'.format(aux_task))), aux_task)

    # load the data for the tasks to be concatenated and concatenate them
    if concat_datasets:
        concat_datasets.sort()
        concat_task_name = '#'.join(concat_datasets)
        logging.info('Loading concat datasets: {}'.format(','.join(concat_datasets)))
        task2data[concat_task_name] = concatenate_datasets(concat_datasets, [utils.load_json(config.get('Files', '{}_datasplit'.format(ds))) for ds in concat_datasets])

    # load the adversarial data (only training data. the data is unlabeled)
    if trg_domain:
        logging.info('Loading (unlabeled) adversarial datasets: {}'.format(trg_domain))
        task2data['adversarial'] = _add_task_indicator(
            utils.load_json(config.get('Files', '{}_unlabeled'.format(trg_domain))), trg_domain)

    return task2data

def _add_task_indicator(data, task):
    for split, ds in data.items():
        if split != 'labelset':
            if 'task' not in data[split].keys():
                data[split]['task'] = [task for i in data[split]['seq']]
    return data



def concatenate_datasets(dataset_names, datasets):
    """
    concatenate the specified datasets
    :param dataset_names:
    :param datasets:
    :return:
    """
    train = {'seq':[], 'label':[], 'task':[]}
    test = {'seq':[], 'label':[], 'task':[]}
    dev = {'seq':[], 'label':[], 'task':[]}
    labelset = []
    for d, split in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
        for ds_name, dataset in zip(dataset_names, datasets):
            if 'task' not in dataset[split].keys():
                d['task'] += [ds_name for i in dataset[split]['seq']]
            for data in dataset[split].keys():
                if data != 'labelset':
                    d[data] += dataset[split][data]
    concat = {'train': train, 'test': test, 'dev':dev}
    labelset = concat['train']['label'] + concat['dev']['label'] + concat['test']['label']
    labelset = list(set(sum(labelset, [])))
    labelset.sort()
    concat['labelset'] = labelset
    concat['train']['labelset'] = labelset
    return concat




def setup_vocabularies(args, task_names, task2data, share_vocabs, wembed_files, setting):
    """
    extracts/loads vocabularies for all the tasks from the respective training datasets.
    some tasks share their vocabulary. if share_vocabs is set,
    all task use one shared vocabulary
    :return:
    """
    # extract the vocabularies from the training sets of all datasets that share the same vocabulary
    embs2task = {}
    if not share_vocabs:
        for tid in task_names:
            embs2task.setdefault(constants.get_emb_name_for_task(tid), []).append(tid)
    else:
        for tid in task_names:
            embs2task.setdefault('shared', []).append(tid)
    # for each vocabulary, collect the data we want to extract it from
    vocabularies = {}

    if setting == 'development':
        for voc, tids in sorted(iter(embs2task.items())):
            logging.info('Extracting vocabulary *{}* for the following datasets:'.format(voc))
            seqs = []
            for tid in tids:
                logging.info('--{}'.format(tid))
                seqs += task2data[tid]['train']['seq']
            vocabularies[voc] = mtl_adversarial.fit_sequences(seqs, wembed_file=wembed_files[voc])
        # save the vocabularies
        for vocab_name, vocab in sorted(iter(vocabularies.items())):
            vocab.save('{}/{}.vocab'.format(args.exp_path, vocab_name))

    elif setting == 'testing':
        # load the vocabularies
        vocabularies = {}
        for voc in sorted(iter(embs2task.keys())):
            logging.info('\nLoading vocabulary *{}*'.format(voc))
            vocab_builder = Word2Int()
            vocab_builder.load('{}/{}.vocab'.format(args.model_dir, voc))
            vocabularies[voc] = vocab_builder
    return vocabularies




def setup_task(task_name, data_name, data, vocab_builder, update_embeds, vocab_name, setting):
    """
    set up the tasks the model is trained/evaluated on. in development setting, we use train/dev/test splits. in testing setting,
    we use train+dev as train and test as test.

    :param task_name:
    :param data_name:
    :param data:
    :param vocab_builder:
    :param update_embeds:
    :param vocab_name:
    :param setting:
    :return:
    """
    input = {'train': {}, 'dev': {}, 'test': {}}
    print(task_name)
    print(data['train'].keys())
    for ds in ['train', 'dev', 'test']:
        input[ds]['seq'] = mtl_adversarial.transform_sequences(data[ds]['seq'], vocab_builder)
        input[ds]['label'] = mtl_adversarial.prepare_labels(data[ds]['label'], data['train']['labelset'])[1]
        input[ds]['labelset'] = data['train']['labelset']
        input['data_name'] = data_name
        if 'task' in data[ds].keys():
            input[ds]['task'] = data[ds]['task']
        else:
            input[ds]['task'] = [task_name for i in range(len(input[ds]['label']))]
    if setting == 'development':
        input_dev = {'train': input['train'], 'dev': input['dev'], 'data_name': input['data_name']}
        trainable = len(input['train']['seq']) > 0
        task = Task(task_name=task_name, data=input_dev, vocab_size=vocab_builder.vocab_size, update_embs=update_embeds,
            vocab_name=vocab_name, trainable=trainable)
        return task
    elif setting == 'testing':
        # merge training and dev datasets
        input_merged = {'train': {'seq': input['train']['seq'] + input['dev']['seq'],
                                  'label': input['train']['label'] + input['dev']['label'],
                                  'labelset': input['train']['labelset']},
                        'task': input['train']['task'] + input['dev']['task'],
                        'dev': {'seq': input['dev']['seq'],
                                'label': input['dev']['label'],
                                'task': input['dev']['task'],
                                'labelset': input['train']['labelset']},
                        'test': {'seq': input['test']['seq'],
                                 'label': input['test']['label'],
                                 'task': input['test']['task'],
                                 'labelset': input['train']['labelset']},
                        'data_name': input['data_name']}
        trainable = len(input['train']['seq']) > 0
        task = Task(task_name=task_name, data=input_merged, vocab_size=vocab_builder.vocab_size,
                    update_embs=update_embeds,
                    vocab_name=vocab_name, trainable=trainable)
        return task

def setup_adversarial_task(task_name, data_name, data, vocab_builder, update_embeds, vocab_name, setting):
    """
    set up the adversarial task. the data for this task is only unlabeled training data

    :param task_name:
    :param data_name:
    :param data:
    :param vocab_builder:
    :param update_embeds:
    :param vocab_name:
    :param setting:
    :return:
    """
    data['train']['seq'] = mtl_adversarial.transform_sequences(data['train']['seq'], vocab_builder)
    data['train']['label'] = []
    data['train']['labelset'] = []
    data['dev'] =  {}
    data['dev']['seq'] = []
    data['dev']['label'] = []
    data['data_name'] = data_name
    print(data.keys())
    task = Task(task_name='adversarial', data=data, vocab_size=vocab_builder.vocab_size, update_embs=update_embeds, vocab_name=vocab_name, trainable=True)
    return task


class Task():

    def __init__(self, task_name, data, vocab_size, update_embs, vocab_name, trainable):
        self.task_name = task_name
        self.data = data
        self.vocab_size = vocab_size
        # the name of the vocabulary used to process the data used for this task
        self.vocab_name = vocab_name
        self.update_embs=update_embs
        self.labelset = data['train']['labelset']
        self.num_classes = len(data['train']['labelset'])
        self.train_seqs = data['train']['seq']
        self.dev_seqs = data['dev']['seq']
        self.train_labels = data['train']['label']
        self.dev_labels = data['dev']['label']
        # in development setting we do not have a test split
        if 'test' in data.keys():
            self.test_seqs = data['test']['seq']
            self.test_labels = data['test']['label']
        self.data_name = data['data_name']

        # indicates if the task has training data or not
        self.trainable = trainable

    def get_params(self):
        params = {}
        params['vocab_size'] = self.vocab_size
        params['update_embs'] = self.update_embs
        params['num_classes'] = self.num_classes
        params['vocab_name'] = self.vocab_name
        params['data_name'] = self.data_name
        params['trainable'] = self.trainable
        return params




def filter_dataset(data, target_labels):
    '''
    filter out all data instances that do not belong to the target classes
    '''
    filtered_data = {}
    target_label_set = set(target_labels)
    # filter out all data points with a label that is not in the set of target labels
    for split_n, split_d in data.items():
        if type(split_d) == dict:
            filtered_split = {'task':[], 'seq':[], 'label':[]}
            for t, s, l in zip(split_d['task'], split_d['seq'], split_d['label']):
                present_target_labels = [elm for elm in l if elm in target_labels]
                if len(present_target_labels) > 0:
                    filtered_split.setdefault('task', []).append(t)
                    filtered_split.setdefault('seq', []).append(s)
                    filtered_split.setdefault('label', []).append(present_target_labels)
            filtered_data[split_n] = filtered_split
    target_labels = list(target_labels)
    target_labels.sort()
    filtered_data['train']['labelset'] = target_labels
    return filtered_data



def empty_result_dict(labelset):
    res = {'p_avg': '-', 'r_avg': '-', 'f_avg': '-','p_macro_avg': '-',
                 'r_macro_avg': '-', 'f_macro_avg': '-'}
    for key in ['p', 'r', 'f']:
        vals = {}
        for label in labelset:
            vals[label] = '-'
        res[key] = vals
    return res


def print_params(args):
    s = 'PARAMETERS:\n'
    for arg in vars(args):
        s += '{}: {}\n'.format(arg, getattr(args, arg))
    return s

def main(args):

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    exp_path = utils.get_exp_path(config, args)
    args.exp_path = exp_path

    ######################################################################################################
    ################################# LOGGING ############################################################
    ######################################################################################################
    # create a logger and set parameters
    logfile = os.path.join(exp_path, '{}.log'.format(args.setting))

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info(print_params(args))


    dynet_params = setup_dynet(random_seed=args.seed, weight_decay=args.weight_decay, mem=args.mem)
    random.seed(args.seed)
    np.random.seed(args.seed)


    wembed_files = {'twitter': '{}/{}'.format(config.get('Dirs', 'glovedir'), 'glove.twitter.27B.{}d.aligned.txt'.format(args.input_dim)),
                    'glove': '{}/{}'.format(config.get('Dirs', 'glovedir'), 'glove.6B.{}d.aligned.txt'.format(args.input_dim)),
                    'shared': ''}



    task2data = load_data(config, args.main_task, args.aux_tasks, args.concat, args.trg_domain)
    task_names = task2data.keys()

    # if the dataset is collapsed, filter the data such that it contains instances with labels that are present in the target task
    if args.eval_set != 'all':
        target_labels = task2data[args.eval_set]['labelset']
        logging.info('Removing instances from datasets that do not correspond to the {} target labels of the target task *{}*'.format(len(target_labels), args.eval_set))
        for task, data in sorted(iter(task2data.items())):
            if task in ['reddit', 'tweets', 'dialogue'] or 'smoking' in task or 'immigration' in task or 'samesex' in task:
                filtered_data = filter_dataset(data, target_labels)
                task2data[task] = filtered_data

    # set up the vocabularies, i.e. extract them from the task data in development setting or load them from disk in test setting
    vocabularies = setup_vocabularies(args, task_names, task2data, setting=args.setting, share_vocabs=args.share_vocabs, wembed_files=wembed_files)

    # set up the tasks
    tasks = {}
    for task_name in task_names:
        if task_name != 'adversarial':
            tasks[task_name] = setup_task(data_name=task_name, task_name=task_name, data=task2data[task_name], vocab_builder=vocabularies[constants.get_emb_name_for_task(task_name)],
                                      update_embeds=args.update_wembeds, vocab_name=constants.get_emb_name_for_task(task_name), setting=args.setting)

    # prepare the adversarial data
    adversarial_task= setup_adversarial_task(data_name=args.trg_domain, task_name='adversarial', data=task2data['adversarial'],
                                             vocab_builder=vocabularies[constants.get_emb_name_for_task(args.trg_domain)],
                                             update_embeds=args.update_wembeds,
                                             vocab_name=constants.get_emb_name_for_task(args.trg_domain), setting=args.setting)


    result_file = '{}/{}.results'.format(exp_path, args.setting)
    pred_file = '{}/{}.predictions'.format(exp_path, args.setting)

    if args.setting=='development':

        # set up the model
        net = mtl_adversarial.MTL_Adversarial_model(num_layers=args.num_layers, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                            vocabularies=vocabularies, update_embs=args.update_wembeds, adversarial_task=adversarial_task, tasks=tasks, main_task=args.main_task, src_domain=args.src_domain,
                            exp_path=exp_path, prediction_layer=args.prediction_layer, additional_params={'dynet_params': dynet_params})

        # train the model and predict the dev set
        validation_data = {'seq': tasks[args.main_task].dev_seqs, 'label': tasks[args.main_task].dev_labels, 'labelset': tasks[args.main_task].labelset}
        final_epoch, final_f_dev, sum_train_losses, no_improvement, best_f_dev = \
            net.train_batched(tasks=tasks, batch_size=args.batch_size, scale_gradient_factor=args.scale_gradient, validation_data=validation_data, seqs_trg=adversarial_task.train_seqs, num_epochs=args.num_epochs, early_stopping=args.early_stopping,
                              patience=args.patience, min_num_epochs=args.min_epochs, num_updates=args.num_updates, prob_main_task=args.prob_main_task, prob_adv=args.prob_adv)
        logging.info('Best model is saved to {}'.format(args.exp_path))
        for tid, task in tasks.items():
            logging.info('Evaluating {}'.format(tid))
            if len(task.train_seqs) == 0:
                preds_train = ['-']
                res_train = empty_result_dict(task.labelset)
            else:
                preds_train = net.predict(tid, task.train_seqs)
                res_train = mtl_adversarial.evaluate_model_predictions(preds_train, task.train_labels, task.labelset)

            preds_dev = net.predict(tid, task.dev_seqs)
            write_model_predictions(preds_dev, pred_file)
            res_dev = mtl_adversarial.evaluate_model_predictions(preds_dev, task.dev_labels, task.labelset)

            log_results(task_name=tid, train=res_train, test=res_dev, labelset=task.labelset, setting=args.setting, result_file=result_file, params=net.model_params,
                        final_epoch=final_epoch, train_loss=sum_train_losses, best_dev_f=best_f_dev)

    elif args.setting=='testing':
        if args.retrain:
            # set up the model
            net = mtl_adversarial.MTL_Adversarial_model(num_layers=args.num_layers, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                                vocabularies=vocabularies, update_embs=args.update_wembeds, tasks=tasks,
                                main_task=args.main_task,
                                exp_path=exp_path,  prediction_layer=args.prediction_layer, additional_params={'dynet_params': dynet_params})
            net.train_batched(tasks=tasks, batch_size=args.batch_size, num_epochs=args.num_epochs, min_num_epochs=args.min_epochs)
        else:
            # load the model trained on train data w/o dev data from disk

            loaded_model_params = utils.load_json('{}/model_params.json'.format(args.model_dir))
            print(loaded_model_params)
            # set up the model
            net = mtl_adversarial.MTL_Adversarial_model(num_layers=loaded_model_params['num_layers'], input_dim=loaded_model_params['input_dim'], hidden_dim=loaded_model_params['hidden_dim'],
                                vocabularies=vocabularies, update_embs=loaded_model_params['update_wembeds'], tasks=tasks,
                                main_task=loaded_model_params['main_task'],
                                exp_path=args.exp_path,
                                prediction_layer=args.prediction_layer,
                                additional_params={'dynet_params': loaded_model_params['dynet_params']}, src_domain=args.src_domain,adversarial_task=adversarial_task)

            net.load(args.model_dir)
        for tid, task in tasks.items():
            logging.info('Evaluating {}'.format(tid))
            if len(task.train_seqs) ==0:
                preds_train = ['-']
                res_train = {'p':'-', 'r': '-', 'f': '-', 'p_avg': '-', 'r_avg': '-', 'f_avg': '-', 'p_macro_avg': '-', 'r_macro_avg': '-', 'f_macro_avg': '-'}

            else:
                preds_train = net.predict(tid, task.train_seqs)
                res_train = mtl_adversarial.evaluate_model_predictions(preds_train, task.train_labels, task.labelset)

            # we predict the validation data here for sanity check
            preds_dev = net.predict(tid, task.dev_seqs)
            res_dev = mtl_adversarial.evaluate_model_predictions(preds_dev, task.dev_labels, task.labelset)

            preds_test = net.predict(tid, task.test_seqs)

            if tid == args.main_task:
                write_model_predictions(preds_test, pred_file)

            res_test = mtl_adversarial.evaluate_model_predictions(preds_test, task.test_labels, task.labelset)

            log_results(task_name=tid, train=res_train, test=res_test, labelset=task.labelset, setting=args.setting, result_file=result_file,
                        params=net.model_params)

def write_model_predictions(preds, pred_file):
    with open(pred_file, 'a') as f:
        for pred in preds:
            f.write('{}\n'.format(pred))
        f.close()

def log_results(task_name, train, test, labelset, setting, result_file, params, final_epoch='testing', train_loss='testing', best_dev_f='testing'):
    with open(result_file, 'a') as f:
        if setting == 'development':
            f.write('Task\tClass\tp_train\tr_train\tf_train\tp_dev\tr_dev\tf_dev\tfinal_epoch\ttrain_loss\tbest_dev_f\tparams\n')
        elif setting == 'testing':
            f.write(
                'Task\tClass\tp_train\tr_train\tf_train\tp_test\tr_test\tf_test\tfinal_epoch\ttrain_loss\tbest_dev_f\tparams\n')
        for label in labelset:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(task_name, label, train['p'][label], train['r'][label], train['f'][label],
                                                                          test['p'][label], test['r'][label], test['f'][label], final_epoch, train_loss, best_dev_f, params))
        # write micro average
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(task_name, 'micro_avg', train['p_avg'], train['r_avg'], train['f_avg'],
                                                                      test['p_avg'], test['r_avg'], test['f_avg'], final_epoch, train_loss, best_dev_f, params))
        # write macro average
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(task_name, 'macro_avg', train['p_macro_avg'],
                                                                          train['r_macro_avg'], train['f_macro_avg'],
                                                                          test['p_macro_avg'], test['r_macro_avg'],
                                                                          test['f_macro_avg'],
                                                                          final_epoch, train_loss, best_dev_f, params))

        f.close()


class ConfigParser():

    def __init__(self, fname):
        self.params = utils.load_json(fname)

    def get(self, sec, key):
        return self.params[sec][key]

    def getint(self, sec, key):
        return int(self.get(sec, key))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Multi-task learning for sentence classification')

    parser.add_argument('--exp_id', type=str, default='',
                        help="Identifier for experiment. Random combination if not specified")
    parser.add_argument('--config', type=str,
                        help="path to the config file")
    parser.add_argument('--seed', type=int, default=1,
                        help="Dynet random seed")
    parser.add_argument('--num_layers', type=int,
                        help="The number of LSTM layers")
    parser.add_argument('--input_dim', type=int, choices= [50,100,200],
                        help="The input dimension")
    parser.add_argument('--hidden_dim', type=int,
                        help="The dimension of the LSTM hidden states")
    parser.add_argument('--num_epochs', type=int,
                        help="The maximum number of training epochs")
    parser.add_argument('--batch_size', type=int,
                        help="The number of instances per batch")
    parser.add_argument('--num_updates', type=int, default=5000,
                        help="The number of updates per epoch")
    parser.add_argument('--setting', type=str, choices=['development', 'testing'],
                        help="Either development (training on train, predicting on dev) or testing (re-training on train+dev, predicting on test)")
    parser.add_argument('--model_dir', type=str, default='',
                        help="The directory the trained model is loaded from")
    parser.add_argument('--early_stopping', action='store_true',
                        help="Do early stopping")
    parser.add_argument('--patience', type=int,
                        help="Patience for early stopping")
    parser.add_argument('--min_epochs', type=int, default=20,
                        help="Minimum number of epochs the model is trained for. Default 20")
    parser.add_argument('--main_task', type=str, choices=constants.DATASETS, required=True,
                        help="The name of the main task (all tasks are evaluated but the main task is used for HP optimization)")
    parser.add_argument('--aux_tasks', nargs='+',
                        choices=constants.DATASETS,
                        help='The names of the auxiliary tasks')
    parser.add_argument('--concat', nargs='+',
                        choices=constants.DATASETS,
                        help='The names of the datasets to be concatenated')

    parser.add_argument('--prediction_layer', type=str, default='main_task',
                        help='The output layer used to predict the main task in case the main task has no training data')
    parser.add_argument('--src_domain', type=str,
                        help='The data used as labeled src domain data in the adversarial task')
    parser.add_argument('--trg_domain', type=str,
                        help='The data used as unlabeled trg domain data in the adversarial task')
    parser.add_argument('--update_wembeds', action='store_true',
                        help="Update the wordembeddings during training")
    parser.add_argument('--update_rnn', action='store_true',
                        help="Update the RNN during training")
    parser.add_argument('--share_vocabs', action='store_true',
                        help="Use pretrained shared wordembeddings for all datasets")
    parser.add_argument('--retrain', action='store_true', default=False,
                        help="Only valid in testing setting. If set, the model is retrained on train+dev data")

    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help="Weight decay parameter")
    parser.add_argument('--mem', default=2048, type=int,
                        help="The amount of memory dynet is accumulating prior to building computation graph")
    parser.add_argument('--eval_set', default='all', type=str, help='The data instances to be evaluated. Only applies if the training data is a joined dataset')

    parser.add_argument('--scale_gradient', default=1., type=float,
                        help='Factor for scaling the loss on the src domain')
    parser.add_argument('--prob_main_task', default=0.5, type=float,
                        help="Probability of sampling the main task")
    parser.add_argument('--prob_adv', default=0.5, type=float,
                        help="Probability of doing an adversarial update")

    args = parser.parse_args()
    main(args)

    '''

    python3 run_dialogue_experiments_mtl.py --num_layers 2 --input_dim 100 --hidden_dim 100 --num_epochs 20 --batch_size 100 --setting development --config ../config/config.cfg --early_stopping --patience 5 --update_wembeds --update_rnn --main_task dialogue --aux_tasks tweets

    '''


