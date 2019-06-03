import numpy as np

def map_to_one_hot_multilabel(annotations, class_labels):
    """
    maps each annotation to a vector of length len(class_labels), that contains 1.0  at position i if class_labels[i] in annotations and 0 otherwise
    """
    one_hots = []
    for anno in annotations:
        one_hot = np.zeros(len(class_labels))
        for i in range(len(class_labels)):
            if class_labels[i] in anno:
                one_hot[i] = 1.0
        one_hots.append(one_hot)
    return np.array(one_hots)

def map_to_cat_multilabel(annotations, class_labels):
    """
    maps each annotation to a list containing the class_labels.index(label) for each label in the annotations
    """
    cats = []
    for anno in annotations:
        cat_list = []
        for label in anno:
            cat_list.append(class_labels.index(label))
        cats.append(cat_list)
    return cats

def map_to_one_hot_binary(annotations, target_label):
    """
    maps each annotation to a one hot vector indicating the presence of the target_label in the annotation
    """
    one_hots = []
    for anno in annotations:
        if target_label in anno:
            one_hots.append([0, 1])
        else:
            one_hots.append([1, 0])
    return np.array(one_hots)

def map_to_cat_binary(annotations, target_label):
    """
    maps each annotation to a value indicating the presence of target_label in the annotation
    """
    cats = []
    for anno in annotations:
        if target_label in anno:
            cats.append(1)
        else:
            cats.append(0)
    return cats