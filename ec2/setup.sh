ROOT_DIR="$(git rev-parse --show-toplevel)"

pip install -U bokeh
pip install -U catboost
pip install -U ggplot
pip install -U ipython
pip install -U jupyter
pip install -U kaggle
pip install -U keras
pip install -U lightgbm
pip install -U matplotlib
pip install -U networkx
pip install -U nltk
pip install -U numpy
pip install -U pandas
pip install -U pip
pip install -U plotly
pip install -U seaborn
pip install -U sklearn
pip install -U tensorflow
pip install -U html5lib
pip install -U tqdm
pip install -U xgboost==0.6a2

pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl 
pip3 install torchvision 
# macOS Binaries dont support CUDA, install from source if CUDA is needed

python -c "import nltk; nltk.download('all')"

mkdir -p $HOME/.kaggle
cp $ROOT_DIR/ec2/kaggle.json $HOME/.kaggle
kaggle competitions download -c competitive-data-science-predict-future-sales
