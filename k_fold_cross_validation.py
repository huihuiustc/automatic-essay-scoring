import math
import random
import copy


def split_dataset_to_fold(total_fold, essay_set_per_category, score_category):
    essay_training_per_fold = []
    essay_test_per_fold = []
    for i in range(total_fold):
        training_current_fold = []
        test_current_fold = []

        total_test_a = int(calculate_total_test_data(i+1, total_fold, len(essay_set_per_category['score_a'])))
        total_test_b = int(calculate_total_test_data(i+1, total_fold, len(essay_set_per_category['score_b'])))

        data = split_dataset(i + 1, total_test_a, essay_set_per_category['score_a'])
        training_current_fold.extend(data['training'])
        test_current_fold.extend(data['test'])

        data = split_dataset(i + 1, total_test_b, essay_set_per_category['score_b'])
        training_current_fold.extend(data['training'])
        test_current_fold.extend(data['test'])

        if score_category >= 3:
            total_test_c = int(calculate_total_test_data(i+1, total_fold, len(essay_set_per_category['score_c'])))
            data = split_dataset(i+1, total_test_c, essay_set_per_category['score_c'])
            training_current_fold.extend(data['training'])
            test_current_fold.extend(data['test'])

            if score_category == 4:
                total_test_d = int(calculate_total_test_data(i+1, total_fold, len(essay_set_per_category['score_d'])))
                data = split_dataset(i+1, total_test_d, essay_set_per_category['score_d'])
                training_current_fold.extend(data['training'])
                test_current_fold.extend(data['test'])

        essay_training_per_fold.append(training_current_fold)
        essay_test_per_fold.append(test_current_fold)

    return {'training_per_fold': essay_training_per_fold, 'test_per_fold': essay_test_per_fold}


def calculate_total_test_data(current_fold, total_fold, total_data):
    total_test_data = 0
    if current_fold <= total_data % total_fold:
        total_test_data = math.floor(total_data / total_fold) + 1
    else:
        total_test_data = math.floor(total_data / total_fold)
    return total_test_data


def split_dataset(fold, total_test_data, essay_list):
    test_list = []

    for i in range(total_test_data):
        attempt = 0
        while True:
            index = random.randint(0, len(essay_list) - 1)
            if essay_list[index].fold == 0:
                essay_list[index].set_fold(fold)
                test_list.append(essay_list[index])
                break
            attempt += 1
            if attempt > 500:
                print "Cannot find unique essay for data test in fold {0}".format(fold)
                exit()

    training_list = copy.deepcopy(essay_list)
    for k in range(len(test_list)):
        for l in range(len(training_list)):
            if test_list[k].essay_id == training_list[l].essay_id:
                training_list.pop(l)
                break

    return {'training': training_list, 'test': test_list}


def reset_assigned_fold(essay_set_per_category, score_category):
    for essay in essay_set_per_category['score_a']:
        essay.set_fold(0)

    for essay in essay_set_per_category['score_b']:
        essay.set_fold(0)

    if score_category >= 3:
        for essay in essay_set_per_category['score_c']:
            essay.set_fold(0)
        if score_category == 4:
            for essay in essay_set_per_category['score_d']:
                essay.set_fold(0)

