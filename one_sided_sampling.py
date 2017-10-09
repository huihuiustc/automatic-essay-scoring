import Essay
import numpy
from xlrd import open_workbook
from preprocessing import *
from term_weighting import *
from k_fold_cross_validation import *
from k_nearest_neighbor import *
import xlsxwriter

data_stopword_path = open("C:\Users\user\Dropbox\Tugas Akhir\Dataset\id.stopwords.02.01.2016.txt", "r")
stopword = data_stopword_path.read().split('\n')
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Bahasa Indonesia SMA 9 Bandung\Dataset.xlsx"
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Sejarah SMA 1 Tumijajar\Dataset Soal Nomor 1.xlsx"
data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Sejarah SMA 1 Tumijajar\Dataset Soal Nomor 4.xlsx"
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Ekonomi SMA 1 Tumijajar\Dataset Soal Nomor 4.xlsx"
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Ekonomi SMA 1 Tumijajar\Dataset Soal Nomor 5.xlsx"


#score_category = 4 #soal bahasa indonesia
score_category = 2 #soal sejarah
#score_category = 3 #soal ekonomi

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

## 2 Class
if len(essay_set_per_category['score_a']) < len(essay_set_per_category['score_b']):
    pos_dataset = copy.deepcopy(essay_set_per_category['score_a'])
    neg_dataset = copy.deepcopy(essay_set_per_category['score_b'])
else:
    pos_dataset = copy.deepcopy(essay_set_per_category['score_b'])
    neg_dataset = copy.deepcopy(essay_set_per_category['score_a'])

print "Start-- Total Positive: {0}, Total Negative: {0}".format(len(pos_dataset), len(neg_dataset))
c_dataset = pos_dataset
index = random.randint(0, len(neg_dataset) - 1)
c_dataset.append(neg_dataset[index])
neg_dataset.pop(index)
print "After Select Random Negative-- Total C: {0}".format(len(c_dataset))


for i in range(len(neg_dataset)):
    data_test = []
    data_test.append(neg_dataset[i])

    term_statistic_list = []
    calculate_term_statistic(c_dataset, term_statistic_list)
    calculate_tf_idf_training(c_dataset, term_statistic_list)
    calculate_tfidf_test(data_test, len(c_dataset), term_statistic_list)
    cosine_similarity_result = calculate_cosine_similarity_tfidf(data_test, c_dataset)

    result_status = one_nn_classification(cosine_similarity_result, c_dataset, data_test)

    if result_status:
        c_dataset.append(neg_dataset[i])
        print "Get Missclassified Data-- Total C: {0}".format(len(c_dataset))

workbook = xlsxwriter.Workbook("New Dataset.xlsx")
data_sheet = workbook.add_worksheet("dataset")

for i in range (len(c_dataset)):
    data_sheet.write(i + 1, 0, c_dataset[i].essay_id)
    data_sheet.write(i + 1, 1, c_dataset[i].essay_value)
    data_sheet.write(i + 1, 2, c_dataset[i].human_rater_score)

workbook.close()

