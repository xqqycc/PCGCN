import argparse

from interface import Interface
from util.conf import ModelConf

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='PCGCN')
args = parser.parse_args()

if __name__ == '__main__':
    # Register your model here
        baseline = ['LightGCN','MF']
        graph_models = ['SGL', 'SimGCL',  'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL','PCGCN','DirectAU']
        sequential_models = []
        print('Baseline Models:')
        print('   '.join(baseline))
        print('-' * 80)
        print('Graph-Based Models:')
        print('   '.join(graph_models))

        print('=' * 80)
        model = args.model_name
        import time
        s = time.time()
        if model in baseline or model in graph_models or model in sequential_models:
            conf = ModelConf('./conf/' + model + '.conf')
        else:
            print('Wrong model name!')
            exit(-1)
        rec = Interface(conf)
        rec.execute()
        e = time.time()
        print("Running time: %f s" % (e - s))
