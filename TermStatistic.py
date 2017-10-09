class TermStatistic(object):
    ni = 0
    freq_a = 0
    freq_b = 0
    freq_c = 0
    freq_d = 0

    def __init__(self, term_content):
        self.term_content = term_content
        self.ni = 1

    def increase_score_distribution(self, human_rater_score):
        if human_rater_score == "A":
            self.freq_a += 1
        elif human_rater_score == "B":
            self.freq_b += 1
        elif human_rater_score == "C":
            self.freq_c += 1
        elif human_rater_score == "D":
            self.freq_d += 1