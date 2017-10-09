class Term(object):
    tf = 0
    ntf = 0
    idf = 0
    chi = 0
    rf = 0
    tf_idf = 0
    tf_chi = 0
    tf_rf = 0

    def __init__(self, term_content):
        self.term_content = term_content

    def set_tf(self, tf):
        self.tf = tf

    def set_ntf(self, ntf):
        self.ntf = ntf

    def set_idf(self, idf):
        self.idf = idf

    def set_chi(self, chi):
        self.chi = chi

    def set_rf(self, rf):
        self.rf = rf

    def set_tf_idf(self, tf_idf):
        self.tf_idf = tf_idf

    def set_tf_chi(self, tf_chi):
        self.tf_chi = tf_chi

    def set_tf_rf(self, tf_rf):
        self.tf_rf = tf_rf
