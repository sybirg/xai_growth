# Explainable machine learning and deep learning approaches to identify metabolic reactions important for bacterial growth under different carbon source conditions

[![DOI]()]

Collection of python source codes for metabolic flux simulation & machine & deep learning used in this study.
For more information, please see the publication: Explainable machine learning and deep learning approaches to identify metabolic reactions important for bacterial growth under different carbon source conditions.

Last update: 2023-08-10

This repository is administered by Hyunjae Woo (whj4851@konkuk.ac.kr), Department of Bioscience and Biotechnology, Konkuk University, Seoul, Republic of Korea.

## 1. Metabolic modelling
The metabolic modelling and simulation code is in [Model_simulation](Model_simulation). To produce the simulated flux data, run "model_simulation.ipynb".

Three reactions were added to the original iML1515 E. coli K-12 genome-scale model for oxaloacetate uptake. The updated model is "iML1515_oaa".

Metabolic simulation of minimization of metabolic adjustment (MOMA) was employed to simulate 1422 gene deletions under 30 carbon conditions.

The simulated output data can be found under [Model_simulation/output](Model_simulation/output).

#### software dependencies
* python 3.7.0
* gurobi 9.1.2
* cobrapy 0.22.1
* numpy 1.21.6
* optlang 1.5.0

## 2. Machine learning
The machine learning code can be found in [Supervised_learning](Supervised_learning). The code can be run through "machine_learning.ipynb".

Elastic-net regression technique (EN) (Zou and Hastie 2005) was employed to train the dataset. 

The coefficients of metabolic reactions from the trained models can be found under [Supervised_learning/EN_output_data](Supervised_learning/EN_output_data).

#### software dependencies
* python 3.6.5
* H2O4GPU 0.2.0
* scikit-learn 0.19.1
* numpy 1.19.5
* pyarrow 6.0.1

## 3. Deep learning
The deep learning code can be found in [Supervised_learning](Supervised_learning). The code can be run through "deep_learning.ipynb".

Multi-layer perceptron (MLP) (Gardner and Dorling 1998) was first employed to train the dataset. Then SHapley Additive exPlanations (SHAP) method (Lundberg and Lee 2017) was used to calculate metabolic feature importance. 

The SHAP values of metabolic reactions from the trained models can be found under [Supervised_learning/MLP_output_data](Supervised_learning/MLP_output_data).

#### software dependencies
* python 3.7.0
* tensorflow 2.7.0
* SHAP 0.41.0
* numpy 1.21.6
* pyarrow 10.0.1
* scikit-learn 1.0.2
