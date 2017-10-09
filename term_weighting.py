import math
import TermStatistic


def calculate_term_statistic(essay_training_list, term_statistic_list):
    for essay in essay_training_list:
        for term in essay.term_list:
            status = False
            if not term_statistic_list:
                status = False

            for term_statistic in term_statistic_list:
                if term.term_content == term_statistic.term_content:
                    status = True
                    selected_term = term_statistic
                    break

            if status:
                selected_term.ni += 1
                selected_term.increase_score_distribution(essay.human_rater_score)
            else:
                t = TermStatistic.TermStatistic(term.term_content)
                t.increase_score_distribution(essay.human_rater_score)
                term_statistic_list.append(t)


def calculate_tf_idf_training(essay_training_list, term_statistic_list):
    total_essay = len(essay_training_list)
    for essay in essay_training_list:
        for term in essay.term_list:
            term_statistic = get_term_statistic(term.term_content, term_statistic_list)
            term.set_idf(math.log(float(total_essay)/float(term_statistic.ni), 10))
            term.set_tf_idf(term.ntf * term.idf)


def calculate_tfchi_tfrf_training(essay_training_list, term_statistic_list):
    total_essay = len(essay_training_list)
    a = 0
    b = 0
    c = 0
    d = 0
    for essay in essay_training_list:
        for term in essay.term_list:
            term_statistic = get_term_statistic(term.term_content, term_statistic_list)
            freq_in_positive_category = count_essay_in_positive_category(essay_training_list, essay.human_rater_score)
            freq_in_negative_category = total_essay - freq_in_positive_category

            if essay.human_rater_score == "A":
                a = float(term_statistic.freq_a)
                b = float(freq_in_positive_category - a)
                c = float(term_statistic.freq_b + term_statistic.freq_c + term_statistic.freq_d)
                d = float(freq_in_negative_category - c)
            elif essay.human_rater_score == "B":
                a = float(term_statistic.freq_b)
                b = float(freq_in_positive_category - a)
                c = float(term_statistic.freq_a + term_statistic.freq_c + term_statistic.freq_d)
                d = float(freq_in_negative_category - c)
            elif essay.human_rater_score == "C":
                a = float(term_statistic.freq_c)
                b = float(freq_in_positive_category - a)
                c = float(term_statistic.freq_a + term_statistic.freq_b + term_statistic.freq_d)
                d = float(freq_in_negative_category - c)
            elif essay.human_rater_score == "D":
                a = float(term_statistic.freq_d)
                b = float(freq_in_positive_category - a)
                c = float(term_statistic.freq_a + term_statistic.freq_b + term_statistic.freq_c)
                d = float(freq_in_negative_category - c)

            chi = (float(total_essay) * float(math.pow((a * d) - (b * c), 2))) / float((a + c) * (b + d) * (a + b) * (c + d))
            rf = math.log(float(2) + (float(a) / float(max(1, c))), 10)
            term.set_chi(chi)
            term.set_tf_chi(term.ntf * chi)
            term.set_rf(rf)
            term.set_tf_rf(term.ntf * rf)


def get_term_statistic(term_content, term_statistic_list):
    term = None
    for term_statistic in term_statistic_list:
        if term_content == term_statistic.term_content:
            term = term_statistic
            break
    return term


def count_essay_in_positive_category(essay_list, human_rater_score):
    count = 0
    for essay in essay_list:
        if essay.human_rater_score == human_rater_score:
            count += 1
    return count


def calculate_tfidf_test(essay_test_list, total_training_essay, term_statistic_list):
    for essay in essay_test_list:
        for term in essay.term_list:
            term_statistic = get_term_statistic(term.term_content, term_statistic_list)
            if term_statistic is None:
                term.set_idf(0)
            else:
                term.set_idf(math.log(float(total_training_essay) / float(term_statistic.ni), 10))

            term.set_tf_idf(term.ntf * term.idf)

def calculate_tfchi_tfrf_test(essay_test_list, essay_training_list):
    for essay in essay_test_list:
        for term in essay.term_list:
            selected_term = get_term_with_max_chi_rf_value(term.term_content, essay_training_list)
            if selected_term['term_max_chi'] is None:
                term.set_chi(0)
            else:
                term.set_chi(selected_term['term_max_chi'].chi)

            if selected_term['term_max_rf'] is None:
                term.set_rf(0)
            else:
                term.set_rf(selected_term['term_max_rf'].rf)

            term.set_tf_chi(term.ntf * term.chi)
            term.set_tf_rf(term.ntf * term.rf)


def get_term_with_max_chi_rf_value(term_content, essay_training_list):
    term_max_chi = None
    term_max_rf = None
    for essay in essay_training_list:
        for term in essay.term_list:
            if term.term_content == term_content:
                if term_max_chi is None:
                    term_max_chi = term
                else:
                    if term.chi > term_max_chi.chi:
                        term_max_chi = term

                if term_max_rf is None:
                    term_max_rf = term
                else:
                    if term.rf > term_max_rf.rf:
                        term_max_rf = term

    return {'term_max_chi': term_max_chi, 'term_max_rf': term_max_rf}
