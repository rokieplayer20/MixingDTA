# MixingDTA
This GitHub repository provides the source code for the MixingDTA[1]. It is intended for academic purposes.

⚠️ Built with ESM

## Abstract

Drug–Target Affinity (DTA) prediction is an important regression task for drug discovery, which can provide richer
information than traditional drug-target interaction prediction as a binary prediction task. To achieve accurate DTA
prediction, quite large amount of data is required for each drug, which is not available as of now. Thus, data scarcity
and sparsity is a major challenge. Another important task is ‘cold-start’ DTA prediction for unseen drug or protein. In
this work, we introduce MixingDTA, a novel framework to tackle data scarcity by incorporating domain-specific pre-
trained language models for molecules and proteins with our MEETA (MolFormer and ESM-based Efficient aggregation
Transformer for Affinity) model. We further address the label sparsity and cold-start challenges through a novel data
augmentation strategy named GBA-Mixup, which interpolates embeddings of neighboring entities based on the Guilt-By-
Association (GBA) principle, to improve prediction accuracy even in sparse regions of DTA space. Our experiments on
benchmark datasets demonstrate that the MEETA backbone alone provides up to a 19% improvement of mean squared
error over current state-of-the-art baseline, and the addition of GBA-Mixup contributes a further 8.4% improvement.
Importantly, GBA-Mixup is model-agnostic, delivering performance gains across all tested backbone models of up to
16.9%. Case studies shows how MixingDTA interpolates between drugs and targets in the embedding space, demonstrating
generalizability for unseen drug–target pairs while effectively focusing on functionally critical residues. These results
highlight MixingDTA’s potential to accelerate drug discovery by offering accurate, scalable, and biologically informed
DTA predictions. 


## Overview of MixingDTA

![MixingDTA](./imgs/MixingDTA.png)
Overview of MixingDTA; a. Two DT pairs are input, and the mixing ratio is sampled from a Beta distribution for training like C-Mixup; b. The default backbone model of MixingDTA is MEETA. It utilizes embeddings from pretrained language models. Cross-AFA efficiently processes through AFA from a computational cost perspective; c. Edges between DT pairs are connected based on the criteria for defining neighbors. Neighbors with similar labels are closer, following the C-Mixup method. For each view, new nodes are augmented between DT nodes, creating mixed embeddings that are then trained; d. This is a step of multi-view integration. The embeddings are extracted from encoders trained on each GBA scenario and fed into the FC layers of MEETA.

⚠️ The term `reversed` used in the code is equivalent to `Complement` as described in the paper.
 

## Prerequisites for running MixingDTA

```
conda create -n MixingDTA python=3.10

pip install transformers
pip install biopython
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install esm # for ESM-3 open small version
pip install ipykernel
pip install PyTDC==1.0.7
pip install easydict
pip install lifelines==0.27.3
pip install scikit-learn==1.2.2
conda install -c conda-forge rdkit=2023.9.6
```

For more detailed environment configuration, refer to `requirements.txt`.

## Processing datasets

The DAVIS and KIBA and BindingDB_Kd datasets were downloaded from [TDC](https://tdcommons.ai/multi_pred_tasks/dti) and processed. They are provided in the `DTA_DataBase` directory of this GitHub repository as pickle files, pre-split for 5-fold cross-validation.
This PDBbind_Refined was obtained from 'https://github.com/MahaThafar/Affinity2Vec/blob/main/PDBBind_Refined/All_PDBbind_info.csv' and subsequently reprocessed by me.



### Extract embeddings from Pre-trained LMs
The embeddings corresponding to Drug SMILES and protein sequences are extracted from pretrained models. The code for extraction is made available. `drug_ids.pkl` and `protein_ids.pkl` serve as inputs for these codes. They are in dictionary format, where the keys are IDs and the values are strings.

```
cd ./DTA_DataBase/MolFormer/make_embeddings.py
python ./make_embeddings.py

cd ./DTA_DataBase/ESM3_open_small/make_embeddings.py
python ./make_embeddings.py

```

For academic details on protein embeddings, refer to [2].

### Google Drive

Due to storage constraints, only the trained parameters for the DeepDTA version of MixingDTA are provided.

```
https://drive.google.com/drive/folders/1rdclHh9DeYL3rcoGP-73cGx2MOn95BLO?usp=drive_link
```


## Training

### Step 1: Training with GBA Perspectives
Refer to part c of the "Overview of MixingDTA" figure.  
The model is trained one by one on six different cases.  

Move to the directory of MEETA, the default backbone model of MixingDTA.
```


python ./main.py --device ${YOUR_GPU_NUM} --case 1 --dataset ${Dataset_name}

python ./main.py --device ${YOUR_GPU_NUM} --case 2 --dataset ${Dataset_name}

python ./main.py --device ${YOUR_GPU_NUM} --case 3 --dataset ${Dataset_name}

python ./main.py --device ${YOUR_GPU_NUM} --case 4 --dataset ${Dataset_name}

python ./main.py --device ${YOUR_GPU_NUM} --case 5 --dataset ${Dataset_name}

python ./main.py --device ${YOUR_GPU_NUM} --case 6 --dataset ${Dataset_name}

```

Dataset_name is one among "DAVIS, KIBA, BindingDB_Kd, PDBbind_Refined".
If you intend to train on the PDBbind_Refined dataset, do not use scenarios 4 and 5 for training.




### Step 2: Meta-Predictor Training
This step is "Multi-View integration".

```
python ./main.py --device ${YOUR_GPU_NUM} --play integration --dataset ${Dataset_name}

```
Including cases progressively from the first perspective up to the sixth leads to improved performance in terms of MSE.  
This experimental result is based on the MSE of the random split on the DAVIS dataset.  
Refer to the "Multi-View Integration" figure below.

![multi-view_integration](./imgs/multi-view_integration.png)
The scenario order follows the sequence described in Section 3.1, "DT Mixup Pair Sampling," of the paper.

## Model-agnostic experiment
The model-agnostic experiments include the simplest model, [DeepDTA](https://github.com/hkmztrk/DeepDTA)[3].  
This is available in the DeepDTA directory, and the input command is identical to the one in the Training section.  


## etc
The version of ESM-3 we used is `esm3-open-small` with 1.4B parameters.  
It is the smallest and fastest model in the ESM-3 family.  
esm3-open-small are provided under the [Cambrian non-commercial license](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement).

<!--
## Licenses
The embeddings and training results derived from esm3-open-small are provided under the [Cambrian non-commercial license](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement). For more details, refer to the following URLs: 
https://github.com/evolutionaryscale/esm?tab=readme-ov-file
https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md

Meanwhile, MolFormer was used to extract embeddings via the Hugging Face API. Refer to [https://huggingface.co/ibm/MoLFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct) for more details. MolFormer is distributed under the Apache License 2.0.
 -->

## References
[1] [MixingDTA] 

[2] [Simulating 500 million years of evolution with a language model]

```
@article{hayes2024simulating,
  title={Simulating 500 million years of evolution with a language model},
  author={Hayes, Tomas and Rao, Roshan and Akin, Halil and Sofroniew, Nicholas J and Oktay, Deniz and Lin, Zeming and Verkuil, Robert and Tran, Vincent Q and Deaton, Jonathan and Wiggert, Marius and others},
  journal={bioRxiv},
  pages={2024--07},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

[3] [DeepDTA]
```
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}
```

