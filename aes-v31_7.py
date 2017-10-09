import Essay
import numpy
from xlrd import open_workbook
from preprocessing import *
from term_weighting import *
from k_fold_cross_validation import *
from k_nearest_neighbor import *
import xlsxwriter
import time

start_time = time.clock()
print "start at %s" % start_time

stopword_path = raw_input("Input lokasi stopword: ")
data_stopword_path = open(stopword_path, "r")
stopword = data_stopword_path.read().split('\n')

data_set_path = raw_input("Input lokasi file dataset: ")
score_category = raw_input("Input jumlah kategori nilai: ")
score_category = int(score_category)

wb = open_workbook(data_set_path)
sheet = wb.sheet_by_index(0)
number_of_rows = sheet.nrows
number_of_column = sheet.ncols

essay_set = []
for row in range(1, number_of_rows):
    values = []
    for col in range(number_of_column):
        value = sheet.cell(row, col).value
        try:
            value = str(int(value))
        except ValueError:
            pass
        finally:
            values.append(value)
    essay = Essay.Essay(*values)
    essay_set.append(essay)


data_set_preprocessing(essay_set, stopword)
print ("Finish load data and preprocessing in --- %s seconds ---" % (time.clock() - start_time))

essay_set_per_category = {'score_a': [], 'score_b': [], 'score_c': [], 'score_d': []}

for essay in essay_set:
    if essay.human_rater_score == "A":
        essay_set_per_category['score_a'].append(essay)
    elif essay.human_rater_score == "B":
        essay_set_per_category['score_b'].append(essay)
    elif essay.human_rater_score == "C":
        essay_set_per_category['score_c'].append(essay)
    elif essay.human_rater_score == "D":
        essay_set_per_category['score_d'].append(essay)

workbook = xlsxwriter.Workbook("Accuracy.xlsx")
accuracy_sheet = []

for i in range(0, 9):
    sheet = workbook.add_worksheet("Akurasi K = {0}".format(i+2))
    accuracy_sheet.append(sheet)

for k in range(2, 11):
    print "============ K-Fold Cross Validation with K = {0} ===========".format(k)
    for i in range(10):
        print "Iterasi {0} Multiple K-Fold".format(i+1)
        reset_assigned_fold(essay_set_per_category, score_category)
        dataset = split_dataset_to_fold(k, essay_set_per_category, score_category)
        essay_training_per_fold = dataset['training_per_fold']
        essay_test_per_fold = dataset['test_per_fold']
        accuracy_sheet[k-2].write(i + 3, 0, "Iterasi {0}".format(i+1))
        accuracy_sheet[k-2].write(i + 19, 0, "Iterasi {0}".format(i + 1))
        accuracy_sheet[k-2].write(i + 35, 0, "Iterasi {0}".format(i + 1))
        accuracy_sheet[k-2].write(i + 51, 0, "Iterasi {0}".format(i + 1))
        accuracy_sheet[k-2].write(i + 67, 0, "Iterasi {0}".format(i + 1))

        for j in range(k):
            essay_training_current_fold = copy.deepcopy(essay_training_per_fold[j])
            essay_test_current_fold = copy.deepcopy(essay_test_per_fold[j])

            term_statistic_list = []
            calculate_term_statistic(essay_training_current_fold, term_statistic_list)
            calculate_tf_idf_training(essay_training_current_fold, term_statistic_list)
            calculate_tfchi_tfrf_training(essay_training_current_fold, term_statistic_list)
            calculate_tfidf_test(essay_test_current_fold, len(essay_training_current_fold), term_statistic_list)
            calculate_tfchi_tfrf_test(essay_test_current_fold, essay_training_current_fold)
            cosine_similarity_result = calculate_cosine_similarity(essay_test_current_fold, essay_training_current_fold)

            print "Fold {0}: TF-IDF".format(j + 1)
            result = knn_classification(cosine_similarity_result['cs_tfidf'], essay_training_current_fold,
                                        essay_test_current_fold, score_category)

            accuracy_sheet[k-2].write(i + 3, j + 1, result['k1_accuracy'])
            accuracy_sheet[k-2].write(i + 19, j + 1, result['k3_accuracy'])
            accuracy_sheet[k-2].write(i + 35, j + 1, result['k5_accuracy'])
            accuracy_sheet[k-2].write(i + 51, j + 1, result['k7_accuracy'])
            accuracy_sheet[k-2].write(i + 67, j + 1, result['k9_accuracy'])

            print "\nFold {0}: TF-CHI".format(j + 1)
            result = knn_classification(cosine_similarity_result['cs_tfchi'], essay_training_current_fold,
                                        essay_test_current_fold, score_category)
            accuracy_sheet[k-2].write(i + 3, j + 13, result['k1_accuracy'])
            accuracy_sheet[k-2].write(i + 19, j + 13, result['k3_accuracy'])
            accuracy_sheet[k-2].write(i + 35, j + 13, result['k5_accuracy'])
            accuracy_sheet[k-2].write(i + 51, j + 13, result['k7_accuracy'])
            accuracy_sheet[k-2].write(i + 67, j + 13, result['k9_accuracy'])

            print "\nFold {0}: TF-RF".format(j + 1)
            result = knn_classification(cosine_similarity_result['cs_tfrf'], essay_training_current_fold,
                                        essay_test_current_fold, score_category)
            accuracy_sheet[k-2].write(i + 3, j + 25, result['k1_accuracy'])
            accuracy_sheet[k-2].write(i + 19, j + 25, result['k3_accuracy'])
            accuracy_sheet[k-2].write(i + 35, j + 25, result['k5_accuracy'])
            accuracy_sheet[k-2].write(i + 51, j + 25, result['k7_accuracy'])
            accuracy_sheet[k-2].write(i + 67, j + 25, result['k9_accuracy'])
            print "\n\n"


workbook.close()
print ("Finish all process in --- %s seconds ---" % (time.clock() - start_time))