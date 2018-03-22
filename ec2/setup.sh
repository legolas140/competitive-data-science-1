ROOT_DIR="$(git rev-parse --show-toplevel)"

pip install kaggle

mkdir -p $HOME/.kaggle
cp $ROOT_DIR/ec2/kaggle.json $HOME/.kaggle
kaggle competitions download -c competitive-data-science-predict-future-sales
