import math
import copy
import Term


def calculate_cosine_similarity(essay_test_list, essay_training_list):
    # Penggambaran Output:
    # 1 Array berisi tiga element yang disebut weightingMethod : TF-IDF, TF-CHI, TF-RF
    # Setiap weightingMethod berisi X elemen yang disebut dataTest, index i merepresentasikan ini data uji keberapa
    # Setiap dataTest berisi Y elemen yang disebut cosineSimilarity, ini menyimpan hasil perhitungan data uji i dengan
    #                                                                       data latih j, j merepresentasikan data latih ke bereapa

    all_test_data_cs_tfidf = []
    all_test_data_cs_tfchi = []
    all_test_data_cs_tfrf = []
    for essay_test in essay_test_list:
        all_train_data_cs_tfidf = []
        all_train_data_cs_tfchi = []
        all_train_data_cs_tfrf = []

        for essay_training in essay_training_list:
            term_list = get_train_test_term_for_cs(essay_test.term_list, essay_training.term_list)
            dot_product = calculate_dot_product(term_list)
            train_vector_length = calculate_vector_length(term_list['train'])
            test_vector_length = calculate_vector_length(term_list['test'])

            #cosine similarity TF-IDF
            if dot_product['tfidf'] == 0 or train_vector_length['tfidf'] == 0 or test_vector_length['tfidf'] == 0:
                cs_tfidf = float(0)
            else:
                cs_tfidf = float(dot_product['tfidf']) / (float(train_vector_length['tfidf']) * float(test_vector_length['tfidf']))

            #cosine similarity TF-CHI
            if dot_product['tfchi'] == 0 or train_vector_length['tfchi'] == 0 or test_vector_length['tfchi'] == 0:
                cs_tfchi = float(0)
            else:
                cs_tfchi = float(dot_product['tfchi']) / (float(train_vector_length['tfchi']) * float(test_vector_length['tfchi']))

            #cosine similarity TF-RF
            if dot_product['tfrf'] == 0 or train_vector_length['tfrf'] == 0 or test_vector_length['tfrf'] == 0:
                cs_tfrf = float(0)
            else:
                cs_tfrf = float(dot_product['tfrf']) / (float(train_vector_length['tfrf']) * float(test_vector_length['tfrf']))

            all_train_data_cs_tfidf.append(cs_tfidf)
            all_train_data_cs_tfchi.append(cs_tfchi)
            all_train_data_cs_tfrf.append(cs_tfrf)

        all_test_data_cs_tfidf.append(all_train_data_cs_tfidf)
        all_test_data_cs_tfchi.append(all_train_data_cs_tfchi)
        all_test_data_cs_tfrf.append(all_train_data_cs_tfrf)

    return {'cs_tfidf': all_test_data_cs_tfidf, 'cs_tfchi': all_test_data_cs_tfchi, 'cs_tfrf': all_test_data_cs_tfrf}


def get_train_test_term_for_cs(test_term_list, train_term_list):
    test_term_content = convert_to_array_string(test_term_list)
    train_term_content = convert_to_array_string(train_term_list)

    cur_train_term_list = []
    for term in test_term_list:
        if term.term_content in train_term_content:
            cur_train_term_list.append(copy.deepcopy(train_term_list[train_term_content.index(term.term_content)]))
        else:
            t = Term.Term(term.term_content)
            t.set_tf_idf(0)
            t.set_tf_chi(0)
            t.set_tf_rf(0)
            cur_train_term_list.append(t)

    cur_test_term_list = copy.deepcopy(test_term_list)
    for term in train_term_list:
        if term.term_content not in test_term_content:
            t = Term.Term(term.term_content)
            t.set_tf_idf(0)
            t.set_tf_chi(0)
            t.set_tf_rf(0)
            cur_test_term_list.append(t)
            cur_train_term_list.append(copy.deepcopy(train_term_list[train_term_content.index(term.term_content)]))
    return {'train': cur_train_term_list, 'test': cur_test_term_list}


def convert_to_array_string(term_list):
    array_string = []
    for term in term_list:
        array_string.append(term.term_content)
    return array_string


def calculate_dot_product(term_list):
    dot_product = {'tfidf': float(0), 'tfchi': float(0), 'tfrf': float(0)}
    train_term_list = term_list['train']
    test_term_list = term_list['test']

    for index in range(len(test_term_list)):
        dot_product['tfidf'] += float(test_term_list[index].tf_idf) * float(train_term_list[index].tf_idf)
        dot_product['tfchi'] += float(test_term_list[index].tf_chi) * float(train_term_list[index].tf_chi)
        dot_product['tfrf'] += float(test_term_list[index].tf_rf) * float(train_term_list[index].tf_rf)

    return dot_product


def calculate_vector_length(term_list):
    vector_length = {'tfidf': float(0), 'tfchi': float(0), 'tfrf': float(0)}
    for index in range(len(term_list)):
        vector_length['tfidf'] += float(math.pow(term_list[index].tf_idf, 2))
        vector_length['tfchi'] += float(math.pow(term_list[index].tf_chi, 2))
        vector_length['tfrf'] += float(math.pow(term_list[index].tf_rf, 2))

    vector_length['tfidf'] = math.sqrt(vector_length['tfidf'])
    vector_length['tfchi'] = math.sqrt(vector_length['tfchi'])
    vector_length['tfrf'] = math.sqrt(vector_length['tfrf'])
    return vector_length


def knn_classification(cosine_similarity_list, essay_training_list, essay_test_list, score_category):
    temp = get_essay_training_id_score(essay_training_list)
    essay_training_id_list = temp['id_list']
    essay_training_score_list = temp['score_list']

    classification_result = []
    for i in range(len(cosine_similarity_list)): #looping untuk memberikan nilai bagi setiap data uji
        essay_test = essay_test_list[i]
        result = {'essay_id': essay_test.essay_id, 'actual_score': essay_test.human_rater_score,
                  'k1_predicted_score': None, 'k3_predicted_score': None, 'k5_predicted_score': None,
                  'k7_predicted_score': None, 'k9_predicted_score': None}

        # buat tuple (cosine similarity, essay_train_id, essay_train_score)
        sorted_data_training = sorted(zip(cosine_similarity_list[i], essay_training_id_list, essay_training_score_list),
                                        reverse=True)
        result['k1_predicted_score'] = sorted_data_training[0][2]
        result['k3_predicted_score'] = k_neighbor_classification(3, sorted_data_training[:3])
        result['k5_predicted_score'] = k_neighbor_classification(5, sorted_data_training[:5])
        result['k7_predicted_score'] = k_neighbor_classification(7, sorted_data_training[:7])
        result['k9_predicted_score'] = k_neighbor_classification(9, sorted_data_training[:9])

        classification_result.append(result)

    return calculate_accuracy(classification_result)


def get_essay_training_id_score(essay_training_list):
    essay_training_id_list = []
    essay_training_score_list = []
    for essay in essay_training_list:
        essay_training_id_list.append(essay.essay_id)
        essay_training_score_list.append(essay.human_rater_score)
    return {'id_list': essay_training_id_list, 'score_list': essay_training_score_list}


def k_neighbor_classification(k, top_k_similar_data):
    list_score = ["A", "B", "C", "D"]
    class_vote = [0, 0, 0, 0]
    for i in range(k):
        human_score = top_k_similar_data[i][2] #struktur tuple: (nilai cosine similarity, essay id, essay score)
        if human_score == "A":
            class_vote[0] += 1
        elif human_score == "B":
            class_vote[1] += 1
        elif human_score == "C":
            class_vote[2] += 1
        elif human_score == "D":
            class_vote[3] += 1

    max_vote = max(class_vote)
    sorted_class_vote = sorted(zip(class_vote, list_score), reverse=True)
    if k == 3:
        if max_vote == 1:
            return top_k_similar_data[0][2]
        else:
            return list_score[class_vote.index(max_vote)]
    if k == 5:
        if max_vote == 2:
            return get_score_with_greater_prior(sorted_class_vote, max_vote, top_k_similar_data)
        else:
            return list_score[class_vote.index(max_vote)]
    if k == 7:
        if max_vote == 2 or max_vote == 3:
            return get_score_with_greater_prior(sorted_class_vote, max_vote, top_k_similar_data)
        else:
            return list_score[class_vote.index(max_vote)]
    if k == 9:
        if max_vote == 3 or max_vote == 4:
            return get_score_with_greater_prior(sorted_class_vote, max_vote, top_k_similar_data)
        else:
            return list_score[class_vote.index(max_vote)]


def get_score_with_greater_prior(sorted_class_vote, max_vote, top_k_similar_data):
    list_possible_score = []
    for i in range (0, len(sorted_class_vote)):
        if sorted_class_vote[i][0] == max_vote:
            list_possible_score.append(sorted_class_vote[i][1])

    if len(list_possible_score) == 1:
        return sorted_class_vote[0][1]
    else:
        score = ""
        for i in range(0,len(top_k_similar_data)):
            if top_k_similar_data[i][2] in list_possible_score:
                score = top_k_similar_data[i][2]
                break
        return score


def k3_neighbor_classification(top_three_similar_data):
    list_score = ["A", "B", "C", "D"]
    class_vote = [0, 0, 0, 0]
    for j in range(3):
        human_score = top_three_similar_data[j][2] #struktur tuple: (nilai cosine similarity, essay id, essay score)
        if human_score == "A":
            class_vote[0] += 1
        elif human_score == "B":
            class_vote[1] += 1
        elif human_score == "C":
            class_vote[2] += 1
        elif human_score == "D":
            class_vote[3] += 1

    max_vote = max(class_vote)
    if max_vote == 1:
        return top_three_similar_data[0][2]
    else:
        return list_score[class_vote.index(max_vote)]


def k5_neighbor_classification(top_five_similar_data):
    list_score = ["A", "B", "C", "D"]
    class_vote = [0, 0, 0, 0]
    for j in range(5):
        human_score = top_five_similar_data[j][2]
        if human_score == "A":
            class_vote[0] += 1
        elif human_score == "B":
            class_vote[1] += 1
        elif human_score == "C":
            class_vote[2] += 1
        elif human_score == "D":
            class_vote[3] += 1

    max_vote = max(class_vote)
    if max_vote == 1:
        return top_five_similar_data[0][2]
    elif max_vote == 2:
        sorted_class_vote = sorted(zip(class_vote, list_score), reverse=True)
        if sorted_class_vote[0][0] == sorted_class_vote[1][0]:
            # Apabila ada 2 nilai yang masing-masing kemunculannya 2x
            for j in range(5):
                score = top_five_similar_data[j][2]
                if score == sorted_class_vote[0][1] or score == sorted_class_vote[1][1]:
                    #ambil nilai yang memiliki kemunculan 2x dan nilai cosine similaritynya tertinggi
                    return score
        else:
            #apabila hanya ada 1 nilai yang kemunculannya 2x
            return list_score[class_vote.index(max_vote)]
    else:
        return list_score[class_vote.index(max_vote)]


def calculate_accuracy(classification_result):
    k1_correct = 0
    k3_correct = 0
    k5_correct = 0
    k7_correct = 0
    k9_correct = 0
    total_data_test = len(classification_result)

    for result in classification_result:
        print "Klasifikasi data uji ID: {0}, Actual score: {1}, 1-NN score: {2}, 3-NN score: {3}, 5-NN score: {4}, " \
              "7-NN score: {5}, 9-NN score: {6}".format(
            result['essay_id'], result['actual_score'], result['k1_predicted_score'], result['k3_predicted_score'],
            result['k5_predicted_score'], result['k7_predicted_score'], result['k9_predicted_score'])

        if result['actual_score'] == result['k1_predicted_score']:
            k1_correct += 1
        if result['actual_score'] == result['k3_predicted_score']:
            k3_correct += 1
        if result['actual_score'] == result['k5_predicted_score']:
            k5_correct += 1
        if result['actual_score'] == result['k7_predicted_score']:
            k7_correct += 1
        if result['actual_score'] == result['k9_predicted_score']:
            k9_correct += 1

    print "Hasil Correct Prediction 1-NN: {0}, 3-NN: {1}, 5-NN: {2}, 7-NN:{3}, 9-NN: {4}, jumlah data uji: {5}".\
        format(k1_correct, k3_correct, k5_correct, k7_correct, k9_correct, total_data_test)

    k1_accuracy = (float(k1_correct) / float(total_data_test)) * 100.0
    k3_accuracy = (float(k3_correct) / float(total_data_test)) * 100.0
    k5_accuracy = (float(k5_correct) / float(total_data_test)) * 100.0
    k7_accuracy = (float(k7_correct) / float(total_data_test)) * 100.0
    k9_accuracy = (float(k9_correct) / float(total_data_test)) * 100.0
    print "Akurasi k1, {0}, k3: {1}, k5: {2}, k7: {3}, k9: {4}".format(k1_accuracy, k3_accuracy ,k5_accuracy, k7_accuracy, k9_accuracy)

    accuracy = {'k1_accuracy': k1_accuracy, 'k3_accuracy': k3_accuracy, 'k5_accuracy': k5_accuracy, 'k7_accuracy': k7_accuracy,
                'k9_accuracy': k9_accuracy}
    return accuracy


#========================= ONE SIDED SAMPLING ================================================#
def calculate_cosine_similarity_tfidf(essay_test_list, essay_training_list):
    essay_test = essay_test_list[0]
    cosine_similarity_result = []
    for essay_training in essay_training_list:
        term_list = get_train_test_term_for_cs(essay_test.term_list, essay_training.term_list)
        dot_product = calculate_dot_product(term_list)
        train_vector_length = calculate_vector_length(term_list['train'])
        test_vector_length = calculate_vector_length(term_list['test'])

        #cosine similarity TF-IDF
        if dot_product['tfidf'] == 0 or train_vector_length['tfidf'] == 0 or test_vector_length['tfidf'] == 0:
            cs_tfidf = float(0)
        else:
            cs_tfidf = float(dot_product['tfidf']) / (float(train_vector_length['tfidf']) * float(test_vector_length['tfidf']))

        cosine_similarity_result.append(cs_tfidf)

    return cosine_similarity_result

def one_nn_classification(cosine_similarity_result, essay_training_list, essay_test_list):
    temp = get_essay_training_id_score(essay_training_list)
    essay_training_id_list = temp['id_list']
    essay_training_score_list = temp['score_list']
    essay_test = essay_test_list[0]

    result = {'essay_id': essay_test.essay_id, 'actual_score': essay_test.human_rater_score,
              'k1_predicted_score': None}

    # buat tuple (cosine similarity, essay_train_id, essay_train_score)
    top_one_similar_data = sorted(zip(cosine_similarity_result, essay_training_id_list, essay_training_score_list),
                                    reverse=True)[:1]

    result['k1_predicted_score'] = top_one_similar_data[0][2]
    print result

    if result['actual_score'] != result['k1_predicted_score']:
        return True
    else:
        return False

#=============================================================================================#
def create_confusion_matrix(classification_result):
    """
    contoh confusion matrix
    [P/A======Nilai A    Nilai B    Nilai C    Nilai D]
    [Nilai A   [0,0]       [0,1]      [0,2]      [0,3]
    [Nilai B   [1,0]       [1,1]      [1,2]      [1,3]
    [Nilai C   [2,0]       [2,1]      [2,2]      [2,3]
    [Nilai D   [3,0]       [3,1]      [3,2]      [3,3]

    Dimana untuk True Positif adalah nilai diagonal
    False Negatif adalah nilai column
    False Positif adalah nilai row
    """
    k3_confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    k5_confusion_matrix = [[0, 0, 0, 0], [0, 0, 0], 0, [0, 0, 0, 0], [0, 0, 0, 0]]
    for result in classification_result:
        actual = result['actual_score']
        k3_predicted = result['k3_predicted_score']
        k5_predicted = result['k5_predicted_score']
        if actual == "A":
            if k3_predicted == "A":
                k3_confusion_matrix[0][0] += 1
            elif k3_predicted == "B":
                k3_confusion_matrix[1][0] += 1
            elif k3_predicted == "C":
                k3_confusion_matrix[2][0] += 1
            else:
                k3_confusion_matrix[3][0] += 1

            if k5_predicted == "A":
                k5_confusion_matrix[0][0] += 1
            elif k5_predicted == "B":
                k5_confusion_matrix[1][0] += 1
            elif k5_predicted == "C":
                k5_confusion_matrix[2][0] += 1
            else:
                k5_confusion_matrix[3][0] += 1
        elif actual == "B":
            if k3_predicted == "A":
                k3_confusion_matrix[0][1] += 1
            elif k3_predicted == "B":
                k3_confusion_matrix[1][1] += 1
            elif k3_predicted == "C":
                k3_confusion_matrix[2][1] += 1
            else:
                k3_confusion_matrix[3][1] += 1

            if k5_predicted == "A":
                k5_confusion_matrix[0][1] += 1
            elif k5_predicted == "B":
                k5_confusion_matrix[1][1] += 1
            elif k5_predicted == "C":
                k5_confusion_matrix[2][1] += 1
            else:
                k5_confusion_matrix[3][1] += 1
        elif actual == "C":
            if k3_predicted == "A":
                k3_confusion_matrix[0][2] += 1
            elif k3_predicted == "B":
                k3_confusion_matrix[1][2] += 1
            elif k3_predicted == "C":
                k3_confusion_matrix[2][2] += 1
            else:
                k3_confusion_matrix[3][2] += 1

            if k5_predicted == "A":
                k5_confusion_matrix[0][2] += 1
            elif k5_predicted == "B":
                k5_confusion_matrix[1][2] += 1
            elif k5_predicted == "C":
                k5_confusion_matrix[2][2] += 1
            else:
                k5_confusion_matrix[3][2] += 1
        else:
            if k3_predicted == "A":
                k3_confusion_matrix[0][3] += 1
            elif k3_predicted == "B":
                k3_confusion_matrix[1][3] += 1
            elif k3_predicted == "C":
                k3_confusion_matrix[2][3] += 1
            else:
                k3_confusion_matrix[3][3] += 1

            if k5_predicted == "A":
                k5_confusion_matrix[0][3] += 1
            elif k5_predicted == "B":
                k5_confusion_matrix[1][3] += 1
            elif k5_predicted == "C":
                k5_confusion_matrix[2][3] += 1
            else:
                k5_confusion_matrix[3][0] += 1

    return {'k3_conf_matrix': k3_confusion_matrix, 'k5_conf_matrix': k5_confusion_matrix}


def performance_evaluation(conf_matrix, score_category):
    tp_a = conf_matrix[0][0]
    fp_a = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3]
    fn_a = conf_matrix[1][0] + conf_matrix[2][0] + conf_matrix[3][0]
    if tp_a == 0:
        precision_a = 0
        recall_a = 0
    else:
        precision_a = (float(tp_a) / float(tp_a + fp_a)) * 100.0
        recall_a = (float(tp_a) / float(tp_a + fn_a)) * 100.0

    tp_b = conf_matrix[1][1]
    fp_b = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3]
    fn_b = conf_matrix[0][1] + conf_matrix[2][1] + conf_matrix[3][1]
    if tp_b == 0:
        precision_b = 0
        recall_b = 0
    else:
        precision_b = (float(tp_b) / float(tp_b + fp_b)) * 100.0
        recall_b = (float(tp_b) / float(tp_b + fn_b)) * 100.0

    if score_category < 3:
        sum_precision = precision_a + precision_b
        sum_recall = recall_a + recall_b
    else:
        tp_c = conf_matrix[2][2]
        fp_c = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3]
        fn_c = conf_matrix[0][2] + conf_matrix[1][2] + conf_matrix[3][2]
        if tp_c == 0:
            precision_c = 0
            recall_c = 0
        else:
            precision_c = (float(tp_c) / float(tp_c + fp_c)) * 100.0
            recall_c = (float(tp_c) / float(tp_c + fn_c)) * 100.0

        if score_category == 4:
            tp_d = conf_matrix[3][3]
            fp_d = conf_matrix[3][0] + conf_matrix[3][1] + conf_matrix[3][2]
            fn_d = conf_matrix[0][3] + conf_matrix[1][3] + conf_matrix[2][3]
            if tp_d == 0:
                precision_d = 0
                recall_d = 0
            else:
                precision_d = (float(tp_d) / float(tp_d + fp_d)) * 100.0
                recall_d = (float(tp_d) / float(tp_d + fn_d)) * 100.0

            sum_precision = precision_a + precision_b + precision_c + precision_d
            sum_recall = recall_a + recall_b + recall_c + recall_d
        else:
            sum_precision = precision_a + precision_b + precision_c
            sum_recall = recall_a + recall_b + recall_c

    macro_avg_precision = 1.0 / (float(score_category) * sum_precision)
    macro_avg_recall = 1.0 / (float(score_category) * sum_recall)
    f_measure = (2 * macro_avg_precision * macro_avg_recall) / (macro_avg_precision + macro_avg_recall)

    return {'macro_avg_precision': macro_avg_precision, 'macro_avg_recall': macro_avg_recall, 'f_measure': f_measure}