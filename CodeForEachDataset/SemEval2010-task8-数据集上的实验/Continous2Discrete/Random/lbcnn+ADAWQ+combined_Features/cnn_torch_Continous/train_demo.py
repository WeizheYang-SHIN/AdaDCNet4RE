import os

from relation_extraction.train import train
from relation_extraction.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    wr_line = open('./output/PRF.txt', 'w', encoding='utf-8')
    wr_line.close()
    train(hparams)


if __name__ == '__main__':
    main()
