# Summarization

Code to Pull datasets from HF datasets, e.g. from  https://huggingface.co/datasets/cnn_dailymail and merge together.
You can inherit from base class (```HFDataset```) to add specific dataset preprocessing.

Some datasets require a manual download and to be placed in the directory (e.g. wikihow)

``` python dataset_info.py``` to start pulling datasets from the fields of the csv and then merge into one dataset.
