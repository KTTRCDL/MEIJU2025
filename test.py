# import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--test', default=False, action=argparse.BooleanOptionalAction)
#     parser.add_argument('--feature_dirs', type=str, nargs='+', default=['path/to/feature1', 'path/to/feature2'],
#                     help='feature directories using list, e.g. --feature_dirs path/to/feature1 path/to/feature2')
#     parser.add_argument('--feature_dims', type=int, nargs='+', default=[1024, 1024],
#                         help='feature dimensions in order using list, e.g. --feature_dims 1024 1024')
#     args = parser.parse_args()
#     # print(args.test)
#     # print(args.feature_dirs)
#     # string to list
#     # args.feature_dirs = eval(args.feature_dirs)
#     print(args.feature_dims, type(args.feature_dims))
#     print(args.feature_dims[0], type(args.feature_dims[0]))

# import argparse

# def modify_opt(opt):
#     opt.value = 42

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--value', type=int, default=0)
#     opt = parser.parse_args()

#     print(f'Before modification: {opt.value}')
#     modify_opt(opt)
#     print(f'After modification: {opt.value}')

import numpy as np
from sklearn.metrics import f1_score
# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]

f1_macro = f1_score(y_true, y_pred, average='macro')

f1_micro = f1_score(y_true, y_pred, average='micro')

f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f'{f1_macro = :.4f} {f1_micro = :.4f} {f1_weighted = :.4f}')