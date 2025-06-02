conda create -n "myenv" python=3.8.0

pip install -r requirements.txt

python -m pip install git+https://github.com/treforevans/uci_datasets.git

chmod +x run_experiments.sh

./run_experiments.sh
