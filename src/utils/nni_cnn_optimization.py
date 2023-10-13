'''
TODO: FILE DA ELIMINARE
import argparse
import nni

from cnn_classifier import main_cnn_optimization

superclasses = [
    ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
    ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
    ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
    ['ACC', 'CHOL', 'LIHC'],
    ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
]

if __name__ == '__main__':
    def get_params():
        parser = argparse.ArgumentParser()
        parser.add_argument("--nf1", type=int, default=100)
        parser.add_argument("--nf2", type=int, default=50)
        parser.add_argument("--nf3", type=int, default=40)
        parser.add_argument("--nf4", type=int, default=30)
        parser.add_argument("--cw1", type=int, default=4)
        parser.add_argument("--cw2", type=int, default=4)
        parser.add_argument("--cw3", type=int, default=4)
        parser.add_argument("--pw1", type=int, default=4)
        parser.add_argument("--pw2", type=int, default=4)
        parser.add_argument("--pw3", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)

        parser.add_argument("--superclass", type=int,
                            help='Please specify the superclass you want to run the experiment '
                                 'for.')
        args, _ = parser.parse_known_args()
        return args

    for i in range(len(superclasses)):
        print(f'RUNNING EXPERIMENT FOR SUPERCLASS {i}')

        try:
            # get parameters form tuner
            tuner_params = nni.get_next_parameter()
            # logger.debug(tuner_params)
            params = vars(get_params())
            params.update(tuner_params)
            main_cnn_optimization(params, i, superclasses[i])
        except Exception as exception:
            # logger.exception(exception)
            raise
'''

