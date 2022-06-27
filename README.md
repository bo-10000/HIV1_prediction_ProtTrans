# HIV1_prediction_ProtTrans

</br> 

This repository is for the prediction of HIV-1 protease cleavage by using pretrained transformer model (ProtTrans)

- **ProtTrans** is pretrained language model for proteins. In this project, we used two models from ProtTrans - **ProtBert, ProtAlbert**.
- We 1) *load the pretrained models* and then 2) *fine-tune them using the HIV-1 protease cleavage dataset.* 
- The code for pretraining is modified from the ProtTrans github (https://github.com/agemagician/ProtTrans)
</br></br>

## Dataset preparation

Please download the dataset from the link below and upzip it under the root (`\`) folder.

- Data download: https://archive.ics.uci.edu/ml/datasets/HIV-1+protease+cleavage

Then the directory structure will be:

```
/newHIV-1_data
   ㄴ1625Data.txt
   ㄴ746Data.txt
   ㄴimpensData.txt
   ㄴschillingData.txt
/dataset.py
/trainer.py
/training_args.yaml
```
## Split train/test data

- There are 4 different datasets in the HIV-1 dataset. We fine-tune the model with one dataset, and test the model with another dataset.
- We split each dataset into train/test dataset (8:2), maintaining the class distribution.
</br></br>

## Fine-tuning & test

You can fine-tune the pretrained model by:
</br></br>
**1. ProtBERT**

```
trainer = HIVTrainer(train_data='746', model_name='Rostlab/prot_bert', training_args_path='training_args.yaml')
trainer.train()

trainer.test(test_data='1625')
```

**2. ProtAlBERT**

```
trainer = HIVTrainer_Albert(train_data='746', model_name='Rostlab/prot_albert', training_args_path='training_args.yaml')
trainer.train()

trainer.test(test_data='1625')
```
</br>

- We tested our code on two models (ProtBert, ProtAlBERT), but you can also try models in ProtTrans. More models can be found at: https://huggingface.co/Rostlab
- You can change training arguments by changing the `training_args.yaml` or create your own .yaml file.
