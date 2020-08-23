import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import argparse
from sklearn.datasets import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')
    parser.add_argument('--dataset_name', type=str, default="iris",
                        help='datasets that are available from: https://scikit-learn.org/stable/datasets/index.html')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='test size to split')
    args = parser.parse_args()

    train_path = f"data/{args.dataset_name}/train.csv"
    test_path = f"data/{args.dataset_name}/test.csv"

    if not os.path.isdir(f"data/{args.dataset_name}"):
        os.makedirs(f"data/{args.dataset_name}", exist_ok=True)
    # download dataset
    if args.dataset_name == "iris":
        data = load_iris()
    elif args.dataset_name == "covtype":
        data = fetch_covtype()
    elif args.dataset_name == "digits":
        data = load_digits()
    elif args.dataset_name == "boston":
        data = load_boston()
    else:
        raise ValueError("NOT supported yet")

    numeric_feature_names = ["f_" + str(i) for i in range(data['data'].shape[1])] if "feature_names" not in data else \
        list(data['feature_names'])

    print(f"numeric_feature_names:\n{numeric_feature_names}")
    label_name = "target"
    dataf = pd.DataFrame(data=np.c_[data['data'], data['target']],
                         columns=numeric_feature_names + ['target'])
    train, test = train_test_split(dataf, test_size=args.test_size)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"done writing {args.dataset_name} to data/{args.dataset_name}")
