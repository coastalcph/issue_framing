import numpy as np

''''
evaluates predictions in a multiclass multilabel setting, i.e. we predict all classes at the same time
and each instance can belong to several classes (gold_annotations)
'''
def evaluate_multiclass_one_vs_all(preds_one_hot, true_labels_one_hot, class_labels):
    # for each class, calculate metrics
    precision = dict()
    recall = dict()
    f = dict()
    stats = dict()
    collector = np.zeros(4)
    for i in range(len(class_labels)):
        class_label = class_labels[i]
        precision[class_label], recall[class_label], f[class_label], stats[class_label] = evaluate_binary(binarize_multi_labels(preds_one_hot, i), binarize_multi_labels(true_labels_one_hot, i))
        for j in range(len(stats[class_label])):
            collector[j] += stats[class_label][j]
    p_avg, r_avg, f_avg = micro_average(collector)
    return {'p':precision, 'r': recall, 'f': f, 'p_avg': p_avg, 'r_avg': r_avg, 'f_avg': f_avg,
            'p_macro_avg': np.mean([precision[c] for c in precision.keys()]),
            'r_macro_avg': np.mean([recall[c] for c in recall.keys()]),
            'f_macro_avg': np.mean([f[c] for c in f.keys()])}

''''
evaluates predictions in a multiclass multilabel setting, i.e. we predict all classes at the same time
and each instance can belong to several classes (gold_annotations)
'''
def evaluate_multiclass(preds_one_hot, true_labels_one_hot, class_labels):
    # for each class, calculate metrics
    precision = dict()
    recall = dict()
    f = dict()
    # 0 tp, 1 fp, 2 fn
    collector = np.zeros((len(class_labels), 4))

    for i in range(len(preds_one_hot)):
        gold = true_labels_one_hot[i]
        gold_classes = [idx for idx in range(len(gold)) if gold[idx] == 1]
        pred = preds_one_hot[i]
        predicted_class = np.argmax(pred)
        for gold_class in gold_classes:
            if predicted_class == gold_class:
                # tp for predicted class
                collector[predicted_class, 0] += 1.
            else:
                # fp for predicted class
                collector[predicted_class, 1] += 1.
                # fn for gold class
                collector[gold_class, 2] += 1.

    for c, class_label in enumerate(class_labels):
        # tp /(tp + fp)
        if (collector[c,0] + collector[c,1]) == 0:
            precision[class_label] = 0
        else:
            precision[class_label] = collector[c,0]/(collector[c,0] + collector[c,1])
        # tp /(tp + fn)
        if (collector[c,0] + collector[c,2]) == 0:
            recall[class_label] = 0
        else:
            recall[class_label] = collector[c, 0] / (collector[c, 0] + collector[c, 2])
        # 2pr / (p+r)
        if (precision[class_label]+recall[class_label]) == 0:
            f[class_label] = 0
        else:
            f[class_label] = 2*precision[class_label]*recall[class_label]/(precision[class_label]+recall[class_label])

    p_avg = np.sum(collector[:,0])/(np.sum(collector[:,0]) + np.sum(collector[:,1]))
    r_avg = np.sum(collector[:, 0]) / (np.sum(collector[:, 0]) + np.sum(collector[:, 2]))
    f_avg = 2*p_avg*r_avg /(p_avg + r_avg)

    return {'p': precision, 'r': recall, 'f': f, 'p_avg': p_avg, 'r_avg': r_avg, 'f_avg': f_avg,
            'p_macro_avg': np.mean([precision[c] for c in precision.keys()]),
            'r_macro_avg': np.mean([recall[c] for c in recall.keys()]),
            'f_macro_avg': np.mean([f[c] for c in f.keys()])}





def binarize_multi_labels(one_hot, target_idx):
    if type(one_hot) == list:
        one_hot= np.array(one_hot)
    return one_hot[:,target_idx]

def generate_random_labels(numLabels, one_hot=False):
    frow = list(np.random.randint(2, size=numLabels))
    if one_hot == False:
        return frow
    else:
        srow = [1 - i for i in frow]
        a = np.zeros((len(frow), 2))
        a[:, 0] = frow
        a[:, 1] = srow
        return a




def evaluate_binary(preds, true_labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(preds)):
        pred = preds[i]
        gold = true_labels[i]
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1
    if (tp + fp) > 0:
        prec = float(tp) / (tp + fp)
    else:
        prec = 0
    if (tp + fn) > 0:
        rec = float(tp) / (tp + fn)
    else:
        rec = 0
    if (prec + rec) > 0:
        f = (2*prec*rec)/(prec + rec)
    else:
        f = 0
    return prec, rec, f, np.array([tp, tn, fp, fn])

def count(target, l):
    return len([i for i in l if i == target])

def micro_average(collector):
    tp = collector[0]
    fp = collector[2]
    fn = collector[3]

    p = tp/(float(tp) + fp)
    r = tp/(float(tp) + fn)
    f = (2*p*r)/(p+r)
    return p, r, f