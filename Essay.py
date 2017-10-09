class Essay(object):
    def __init__(self, essay_id, essay_content, human_rater_score):
        self.essay_id = essay_id
        self.essay_content = essay_content
        self.human_rater_score = human_rater_score
        self.fold = 0

    def set_term_list(self, term_list):
        self.term_list = term_list

    def set_fold(self, fold):
        self.fold = fold

    def show_term_list(self):
        str = "Essay {0}\nTerm : [".format(self.essay_id)
        for term in self.term_list:
            str += "'{0}', ".format(term.term_value)
        str += ']'
        print str