from dataset import CSVDataset
from mlp import MLP
from trainer import Trainer

# here we use the test set that is simply sampled 10% from the original iris dataset.
train_path = "./data/iris/train.csv"
test_path = "./data/iris/test.csv"

save_path = f"tmp/mlp_iris"  # where the model's checkpoints are saved to

numeric_feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# load train data and dev data
train_data = CSVDataset(train_path, numeric_feature_names=numeric_feature_names, label_name="target")
dev_data = CSVDataset(test_path, numeric_feature_names=numeric_feature_names, label_name="target")
# initialize a model
model = MLP(len(numeric_feature_names), train_data.num_label, hidden_units=[64, 32, 16], device="cpu")
# initialize a trainer
trainer = Trainer(train_data, model,
                  dev_data=dev_data,
                  eval_on="accuracy",
                  loss_fn="ce",
                  eval_every=2,
                  device="cpu",
                  save_path=save_path,
                  train_epochs=100,
                  train_batch_size=128)
# start training
trainer.train()

#### load model from the save path
# this can be run separately from training
print("load from saved path")
model = MLP.load_model(save_path)
print("*******************eval on train set of iris*******************")
print(model.evaluate(train_data, device="cpu"))
print("*************eval on test set of iris*******************")
print(model.evaluate(dev_data, device="cpu"))