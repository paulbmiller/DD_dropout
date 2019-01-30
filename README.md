Branch of the DeepDIVA framework to try different aggregation methods at test time of thinned networks created by Dropout.

The updated framework is available at https://github.com/DIVA-DIA/DeepDIVA.

The main additions are contained in the class dropout_testing to analyze test-time Dropout. Running it will output a text file in the log folder which contains some information such as the right predictions of the dropout method, the mean, the median, the number of conflicting networks, etc. For my final results, I have averaged this information over the 20 runs by using Excel spreadsheets which are in the archive results/results.zip.

I have also added two arguments to choose the number of forward passes at test time with Dropout active and the aggregation method.

Typical run example:
python template/RunMe.py --runner-class dropout_testing --output-folder log --dataset-folder datasets/MNIST --lr 0.1 --model-name alexnet --ignoregit --epochs 20 --dropout-samples 100 --aggr "scoring"
