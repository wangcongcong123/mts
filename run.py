from dataset import CSVDataset
from mlp import MLP
from trainer import Trainer
import pandas as pd
import argparse


def get_numeric_feature_names(dataset_path):
    df = pd.read_csv(dataset_path)
    feature_names = list(df.columns)
    # num_label=len(set(list(df["target"])))
    feature_names.remove("target")
    return feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')

    parser.add_argument('--dataset_name', type=str, default="iris",
                        help='datasets that are in ./data after obtaining and splitting by obtain_split_data.py')
    parser.add_argument('--task', type=str, default="cls",
                        help='this is a classification (cls) task or regression (reg) task?')
    parser.add_argument('--train_epochs', type=int, default=100,
                        help='train epochs')
    parser.add_argument('--train_batch_size', type=int, default=130,
                        help='train batch size')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu (cuda) or cpu (cpu)?')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')

    parser.add_argument(
        "--do_train", action="store_true", help="Do training"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Eval test"
    )

    args = parser.parse_args()
    args.do_train = True
    args.do_eval = True

    train_path = f"data/{args.dataset_name}/train.csv"
    test_path = f"data/{args.dataset_name}/test.csv"
    save_path = f"tmp/mlp_{args.dataset_name}"

    numeric_feature_names = get_numeric_feature_names(train_path)

    train_data = CSVDataset(train_path, numeric_feature_names=numeric_feature_names, label_name="target",is_reg=args.task=="reg")
    dev_data = CSVDataset(test_path, numeric_feature_names=numeric_feature_names, label_name="target",is_reg=args.task=="reg")

    if args.do_train:
        model = MLP(len(numeric_feature_names), train_data.num_label if args.task == "cls" else 1, task=args.task,
                    hidden_units=[128,64, 32],device=args.device)
        trainer = Trainer(train_data, model, dev_data=dev_data, eval_on="accuracy" if args.task == "cls" else "loss",
                          loss_fn="ce" if args.task == "cls" else "mse", save_path=save_path, eval_every=2,device=args.device,
                          train_epochs=args.train_epochs, train_batch_size=args.train_batch_size, lr=args.lr)
        trainer.train()

    if args.do_eval:
        print("load from saved path")
        model = MLP.load_model(save_path)
        print(f"*******************eval on train set of {args.dataset_name}*******************")
        print(model.evaluate(train_data,criterion_name="ce" if args.task == "cls" else "mse",device=args.device))
        print(f"*************eval on test set of {args.dataset_name}*******************")
        print(model.evaluate(dev_data,criterion_name="ce" if args.task == "cls" else "mse",device=args.device))
