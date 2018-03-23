ROOT_DIR="$(git rev-parse --show-toplevel)"

pip install -U catboost
pip install -U ipython
pip install -U jupyter
pip install -U kaggle
pip install -U keras
pip install -U lightgbm
pip install -U matplotlib
pip install -U nltk
pip install -U numpy
pip install -U pandas
pip install -U pytorch
pip install -U matplotlib
pip install -U seaborn
pip install -U sklearn
pip install -U tensorflow
pip install -U html5lib
# pip install -U xgboost

python -c "import nltk; nltk.download('all')"

mkdir -p $HOME/.kaggle
cp $ROOT_DIR/ec2/kaggle.json $HOME/.kaggle
kaggle competitions download -c competitive-data-science-predict-future-sales
