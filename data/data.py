class Data(object):
    def __init__(self, conf, training_set_IF, test_set,training_set_IU,test_IF,test_IU):
        self.config = conf
        self.training_data = training_set_IF[:]
        self.test_data = test_set[:]
        self.implicit_data= training_set_IU[:]
        self.training_data_total=self.training_data+self.implicit_data
        self.test_high_data=test_IF
        self.test_low_data=test_IU






