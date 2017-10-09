import Essay
import numpy
from xlrd import open_workbook
from preprocessing import *
from term_weighting import *
from k_fold_cross_validation import *
from k_nearest_neighbor import *
import xlsxwriter


#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Bahasa Indonesia SMA 9 Bandung\Dataset - ORI.xlsx"
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Sejarah SMA 1 Tumijajar\Dataset Soal Nomor 1 - ORI.xlsx"
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Sejarah SMA 1 Tumijajar\Dataset Soal Nomor 4 - ORI.xlsx"
#data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Ekonomi SMA 1 Tumijajar\Dataset Soal Nomor 4 - ORI.xlsx"
data_set_path = "C:\Users\user\Dropbox\Tugas Akhir\Dataset\Essay Ekonomi SMA 1 Tumijajar\Dataset Soal Nomor 5 - ORI.xlsx"


#score_category = 4 #soal bahasa indonesia
#score_category = 2 #soal sejarah
score_category = 3 #soal ekonomi

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

a_dataset = raw_input("Input jumlah dataset A: ")
a_dataset = int(a_dataset)
b_dataset = raw_input("Input jumlah dataset B: ")
b_dataset = int(b_dataset)

final_dataset = []
for i in range(0, a_dataset):
    random_index = random.randint(0, len(essay_set_per_category['score_a']) - 1)
    final_dataset.append(essay_set_per_category['score_a'][random_index])
    essay_set_per_category['score_a'].pop(random_index)

for i in range(0, b_dataset):
    random_index = random.randint(0, len(essay_set_per_category['score_b']) - 1)
    final_dataset.append(essay_set_per_category['score_b'][random_index])
    essay_set_per_category['score_b'].pop(random_index)

if score_category >= 3:
    c_dataset = raw_input("Input jumlah dataset C: ")
    c_dataset = int(c_dataset)
    for i in range(0, c_dataset):
        random_index = random.randint(0, len(essay_set_per_category['score_c']) - 1)
        final_dataset.append(essay_set_per_category['score_c'][random_index])
        essay_set_per_category['score_c'].pop(random_index)

    if score_category == 4:
        d_dataset = raw_input("Input jumlah dataset D: ")
        d_dataset = int(d_dataset)
        for i in range(0, d_dataset):
            random_index = random.randint(0, len(essay_set_per_category['score_d']) - 1)
            final_dataset.append(essay_set_per_category['score_d'][random_index])
            essay_set_per_category['score_d'].pop(random_index)


workbook = xlsxwriter.Workbook("New Dataset.xlsx")
data_sheet = workbook.add_worksheet("dataset")

for i in range (len(final_dataset)):
    data_sheet.write(i + 1, 0, final_dataset[i].essay_id)
    data_sheet.write(i + 1, 1, final_dataset[i].essay_value)
    data_sheet.write(i + 1, 2, final_dataset[i].human_rater_score)

workbook.close()