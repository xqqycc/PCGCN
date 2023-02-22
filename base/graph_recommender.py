from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set_IF, test_set,training_set_IU,test_IF,test_IU, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set_IF, test_set,training_set_IU,test_IF,test_IU, **kwargs)
        self.data = Interaction(conf, training_set_IF, test_set,training_set_IU,test_IF,test_IU)#data是类
        self.bestPerformance = []
        self.bestPerformance_high = []
        self.bestPerformance_low = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)#candidates里的是编号
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)#rated_list里是名字
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8#故此处要把名字找到对应的编号
            ids, scores = find_k_largest(self.max_N, candidates)#ids也是编号
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list#rec_list里的item是名字

    def high_test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_high_set)
        for i, user in enumerate(self.data.test_high_set):
            candidates = self.predict(user)#candidates里的是编号
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)#rated_list里是名字
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8#故此处要把名字找到对应的编号
            ids, scores = find_k_largest(self.max_N, candidates)#ids也是编号
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list#rec_list里的item是名字

    def low_test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_low_set)
        for i, user in enumerate(self.data.test_low_set):
            candidates = self.predict(user)#candidates里的是编号
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)#rated_list里是名字
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8#故此处要把名字找到对应的编号
            ids, scores = find_k_largest(self.max_N, candidates)#ids也是编号
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list#rec_list里的item是名字

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.output['-dir']
        file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        # high_result = ranking_evaluation(self.data.test_high_set, rec_list, self.topN)
        # low_result = ranking_evaluation(self.data.test_low_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        # self.model_log.add('###High Evaluation Results###')
        # self.model_log.add(high_result)
        # self.model_log.add('###Low Evaluation Results###')
        # self.model_log.add(low_result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))
        # print('The result of %s for high:\n%s' % (self.model_name, ''.join(high_result)))
        # print('The result of %s for low:\n%s' % (self.model_name, ''.join(low_result)))

    def fast_evaluation(self, epoch):
        print('evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])#['Top 20\n', 'Hit Ratio:0.02820386847899267\n', 'Precision:0.0051158492309723415\n', 'Recall:0.03226748256813314\n', 'NDCG:0.01756549743290297\n']
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Quick Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + ' | '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure

    def high_evaluation(self, epoch):
        print('evaluating the model...')
        rec_list = self.high_test()
        measure = ranking_evaluation(self.data.test_high_set, rec_list, [
            self.max_N])  # ['Top 20\n', 'Hit Ratio:0.02820386847899267\n', 'Precision:0.0051158492309723415\n', 'Recall:0.03226748256813314\n', 'NDCG:0.01756549743290297\n']
        if len(self.bestPerformance_high) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance_high[1]:
                if self.bestPerformance_high[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance_high[1] = performance
                self.bestPerformance_high[0] = epoch + 1
        else:
            self.bestPerformance_high.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance_high.append(performance)
        print('-' * 120)
        print('High Rating Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance_high[1]['Hit Ratio']) + ' | '
        bp += 'Precision' + ':' + str(self.bestPerformance_high[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(self.bestPerformance_high[1]['Recall']) + ' | '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance_high[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance_high[0]) + ',', bp)
        print('-' * 120)
        return measure


    def low_evaluation(self, epoch):
        print('evaluating the model...')
        rec_list = self.low_test()
        measure = ranking_evaluation(self.data.test_low_set, rec_list, [
            self.max_N])  # ['Top 20\n', 'Hit Ratio:0.02820386847899267\n', 'Precision:0.0051158492309723415\n', 'Recall:0.03226748256813314\n', 'NDCG:0.01756549743290297\n']
        if len(self.bestPerformance_low) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance_low[1]:
                if self.bestPerformance_low[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance_low[1] = performance
                self.bestPerformance_low[0] = epoch + 1
        else:
            self.bestPerformance_low.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance_low.append(performance)
        print('-' * 120)
        print('Low Rating Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance_low[1]['Hit Ratio']) + ' | '
        bp += 'Precision' + ':' + str(self.bestPerformance_low[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(self.bestPerformance_low[1]['Recall']) + ' | '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance_low[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance_low[0]) + ',', bp)
        print('-' * 120)
        return measure

