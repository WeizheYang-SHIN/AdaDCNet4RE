import os
import argparse

here = os.path.dirname(os.path.abspath(__file__))
# default_pretrained_model_path = os.path.join(here, '../pretrained_models/bert-base-chinese')
default_pretrained_word_vectors_path = os.path.join(here, '../pretrained_vectors/GoogleNews-vectors-negative300.bin')
default_train_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json')
default_validation_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json')
default_test_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/process_TEST_FILE_FULL.json')
default_output_dir = os.path.join(here, '../saved_models')
default_log_dir = os.path.join(default_output_dir, 'runs')
default_tagset_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/tags.txt')
default_model_file = os.path.join(default_output_dir, 'model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')
parser = argparse.ArgumentParser()

# nInputPlane=1, numChannels=8, numWeights=16, full=50, depth=2, sparsity=0.5,
# parser.add_argument("--inInputPlane",type=int,default=1)
# parser.add_argument("--numChannels",type=int,default=3)
parser.add_argument("--numWeights",type=int,default=32)
parser.add_argument("--full",type=int,default=36)
parser.add_argument("--depth",type=int,default=6)
parser.add_argument("--sparsity",type=int,default=0.5)

# parser.add_argument("--pretrained_model_path", type=str, default=default_pretrained_model_path)
parser.add_argument("--pretrained_word_vectors",type=str,default=default_pretrained_word_vectors_path)
parser.add_argument("--train_file", type=str, default=default_train_file)
parser.add_argument("--validation_file", type=str, default=default_validation_file)
parser.add_argument("--test_file", type=str, default=default_test_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--tagset_file", type=str, default=default_tagset_file)
parser.add_argument("--model_file", type=str, default=default_model_file)
parser.add_argument("--checkpoint_file", type=str, default=default_checkpoint_file)

# model
parser.add_argument('--embedding_dim', type=int, default=300, required=False, help='embedding_dim')
parser.add_argument('--dropout', type=float, default=0.05, required=False, help='dropout') # 0.05 配上 1e-4 = 64.00   0配上1e-4=62.47
# parser.add_argument('--stride', type=int, default=2)

# parser.add_argument('--device',type=str,default='"cuda" if torch.cuda.is_available() else "cpu"')
parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--device', type=str, default="cpu")
parser.add_argument("--seed", type=int, default=12360)
parser.add_argument("--max_len", type=int, default=148)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--validation_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--kernel_size1", type=int, default=7)
parser.add_argument("--kernel_size2", type=int, default=7)
parser.add_argument("--kernel_size3", type=int, default=7)
parser.add_argument("--kernel_size4", type=int, default=7)
parser.add_argument("--kernel_size5", type=int, default=7)
parser.add_argument("--kernel_size6", type=int, default=7)
parser.add_argument("--kernel_size7", type=int, default=7)
parser.add_argument("--kernel_size8", type=int, default=7)
parser.add_argument("--kernel_size9", type=int, default=7)
# parser.add_argument("--kernel_size10", type=int, default=7)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--learning_rate", type=float, default=1e-4) # 1e-5 58.54 1e-4 62.21 2e-5 59.77 1e-3 58.03
parser.add_argument("--weight_decay", type=float, default=0.001) # 权重衰减，减少过拟合问题

parser.add_argument('--num_words', type=int, default=2500)
parser.add_argument("--out_dim", type=int, default=30)
hparams = parser.parse_args()
