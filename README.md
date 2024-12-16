# Milestone description
This milestone focuses on the development of Graph Neural network models used in the domain of chemical property prediction. The focus will be on applying the model to a binary classification task on the [hiv dataset](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) and a regression task on the [lipo dataset](https://ogb.stanford.edu/docs/graphprop/) and comparing them to more traditional Machine learning methods such as Random Forests (RF) and Logistic Regression (LR). For better understanding, we will briefly describe the architecture of the whole project, so the reader will be familiar with how the current application fits into the bigger picture. Afterward, we will present the structure of the code and repository, the tools and processes used, and finally the development, evaluation, and comparison of the already mentioned models.

## Proposed application architecture
The application follows a microservice architecture, which makes it easier to add new machine-learning (ML) models and also enables us to scale different parts independently. In the image below we can see the main components of our application
<img src="https://github.com/user-attachments/assets/1410194d-84f6-433e-92b9-004f618eb7f2" 
     style="display:block;float:none;margin-left:auto;margin-right:auto;width:60%"> 

**Toxic Frontend** is a Svelte application that defines the WebUI of the application and enables the user to interact with the system. **ToxicBackend** is a Kotlin [spring boot](https://spring.io/projects/spring-boot) application, which communicates with the database and ML microservices through a REST interface and sends the final results to the frontend service for display. It also defines a common format that the ML services need to implement for them to be compatible with the system. This allows us to add multiple different ML models as long as we can expose their functionality through the pre-defined interface. The final component of the system are the ML models, which can be enterprise software such as [Vega](https://www.vegahub.eu/portfolio-types/in-silico-models/) or ToxicMl which was developed specifically for this application and is the focus of this report. 

## ToxicML Project structure
The project has the following file structure
```text
📂 ToxicML:
|_📂 dataset
|_📂 notebooks
|_📂 saved_models
|_📂 tests
|_📂 ToxicMl
  |_📂 Api
    |_📄 main.py
  |_📂 Mlmodels
    |_📄 preprocessing.py
    |_📄 ...
  |_📄 dataset.py
  |_📄 evaluation.py
  |_📄 metrics.py
  |_📄 trainer.py
📄 Dockerfile
```
- **dataset** contains all of the datasets used in the project.
- **notebooks** contain Jupyter notebooks that have been used to run ML experiments. Normally scripts are preferred, but in this case, I decided to use notebooks for reproducibility
- **saved_models** contains a subset of the best models
- **test** contains multiple [pytest](https://docs.pytest.org/en/stable/) files which are executed as part of the CI/CD pipeline described in the upcoming section.
- **ToxicMl/Api/main.py** is the entry point of the application. When run it starts a [FastApi](https://fastapi.tiangolo.com/) and exposes a swagger interface on localhost:8081/docs. Currently, the developed models are not yet integrated into the API
- **Mlmodels/preprocessing.py** Contains utility functions that are used to convert SMILE codes to [PyG graphs](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or generate graph level features which can be used by both deep and traditional ML methods
- **Mlmodels/..** This directory also contains all of the NN architecture code of all of our models
- **trainer.py** Contains classes used for training regression and classification problems and logging their performance to [Weights & Biases](https://wandb.ai/site)
- **metrics.py** Contains additional custom metrics used evaluation of our models
- **dataset.py** Contains [PyG dataset objects](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.data.InMemoryDataset.html) used for the loading and preparing of datasets
## Dependency management
Instead of the traditional requirements.txt file, this project uses [Poetry](https://python-poetry.org/) for dependency management since it offers superior dependency management for Python projects by providing a lock file (poetry.lock) that ensures repeatable builds by tracking precise versions of all dependencies, including transitive ones. It has a better dependency resolver than pip and helpful error messages. It also removes the need for multiple configuration files (e.g., setup.py, setup.cfg, and test configs) which reduces repository clutter. Additionally, it simplifies virtual environment management and streamlines building and publishing packages to PyPI.

## CI/CD
For Continues integration and development this project relies on [GitHub Actions](https://github.com/features/actions). We have defined a simple pipeline that contains the following steps:
1. setup Python and poetry
2. install dependencies using poetry
3. run pytest
4. code quality check with [ruff](https://github.com/astral-sh/ruff-action)
5. build docker container
6. publish container to [DockerHub](https://hub.docker.com/repository/docker/custibor29/toxicml/general)

If any of the steps fail the whole pipeline fails
## Modeling
The Hiv dataset contains 41128 data points, of which only 1443 represent positive classes. We will focus on improving the f1-score of our classifier and try to reach a performance similar to that of traditional ML methods. 

The lipo dataset is small for a deep-learning task. It contains only 4000 data points. The target values range from -1.5 to 4.5. The focus will be on optimizing the mean squared error of our model, and we will try to achieve a similar or better score than the traditional methods. 

Both datasets already come pre-split, which makes it easy to compare different models performances. 

### Traditional ML methods
Before using the traditional methods, we need to do feature engineering. Currently, both of the datasets only contain two different variables the target variable which we are trying to predict and the [SMILE](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System) encoding of the variable, which on its own cannot be used as a feature for classical ml algorithms, that work on tabular values. We can use Rdkit, a Python library for cheminformatics, to compute different numerical features of the chemicals in our dataset. With that, we can construct a 210-dimensional feature vector, which will be used during model fitting. For each dataset, we will build at least two models and present their non-optimized (baseline) performance and their performance with optimized hyperparameters.

#### Performance and evaluation
| model                         | f1 score | precision | recall   |
|-------------------------------|----------|-----------|----------|
| Logistic regression           | 0.243902 | 0.588235  | 0.153846 |
| Random forest                 | 0.225166 | **0.809524**  | 0.130769 |
| Naive bayes                   | 0.080503 | 0.042572  | 0.738462 |
| Logistic regression optimized | 0.140713 | 0.080128  | 0.576923 |
| **Random Forest optimized**   | **0.275000** | 0.733333  | 0.169231 |
| Naive bayes optimized         | 0.096937 | 0.051659  | **0.784903** |


| model                        | mean absolute error | mean squared error | max error    |
|------------------------------|---------------------|--------------------|--------------|
| Lasso                        | 1.006285            | 1.484697           | **3.261751** |
| Ridge                        | 0.683129            | 0.935416           | 9.890663     |
| Random Forrest               | 0.614670            | 0.629434           | 3.525267     |
| Lasso optimized              | 0.695592            | 0.910692           | 8.294026     |
| Ridge optimized              | 0.679734            | 0.926124           | 9.812714     |
| **Random Forrest optimized** | **0.609851**        | **0.619291**       | 3.591070     |



### Graph Neural Networks
#### Graph Encoding
#### Overview of Graph Methods
#### Establishing a baseline

| model                   | f1      | precision | recall  |
|-------------------------|---------|-----------|---------|
| base GNN                | 0       | 0         | 0       |
| base GNN, weighted loss | 0       | 0         | 0       |
| base GNN, sampling      | 0.05941 | 0.4815    | 0.03166 |


| model    | mean absolute error | mean squared error | max error |
|----------|---------------------|--------------------|-----------|
| base GNN | 1.048               | 1.63               | 3.408     |


#### Trying Different model architectures

|model         |F1               |Recall          |Precision       |
|--------------|-----------------|----------------|----------------|
|ATTENTION 3-32|0.0724174653887114|0.0396270396270396|0.419753086419753|
|ATTENTION 3-64|0.0693069306930693|0.0376749192680301|0.432098765432099|
|ATTENTION 5-64|0.0650835532102023|0.0350378787878788|0.45679012345679|
|ATTENTION 5-32|0.0549717057396928|0.0294117647058823|0.419753086419753|
|SAGE 5-32     |0.0653386454183267|0.034923339011925|0.506172839506173|
|SAGE 3-64     |0.0577777777777778|0.0317460317460317|0.320987654320988|
|SAGE 3-32     |0.0572226099092812|0.0303254437869822|0.506172839506173|
|SAGE 5-64     |0.0556038227628149|0.0299065420560748|0.395061728395062|
|GCN  3-64     |0.0717009916094584|0.0382113821138211|0.580246913580247|
|GCN  5-64     |0.0692751763951251|0.0365358592692828|0.666666666666667|
|GCN  5-32     |0.0654911838790932|0.0345056403450564|0.641975308641975|
|GCN  3-32     |0.0629241209130166|0.0331168831168831|0.62962962962963|



|Model         |mean absolute error |mean squared error|Max Error  |
|--------------|-----------------|----------------|----------------|
|SAGE 5-64     |0.996519923210144|1.56879380544027|7.54553270339966|
|SAGE 3-32     |1.01025628702981 |1.5882872626895 |3.75796175003052|
|SAGE 3-16     |1.03492550736382 |1.63926373436337|5.46569204330444|
|SAGE 5-32     |1.04846721149626 |1.88859879629953|10.9541702270508|
|GCN 3-32      |0.983594309715997|1.48278082893008|3.87179160118103|
|GCN 5-64      |1.00551666078113 |1.50818571817307|3.76725649833679|
|GCN 3-16      |1.0352318854559  |1.55618706657773|3.76055002212524|
|GCN 5-32      |1.02038952736627 |1.58759326934814|3.82405233383179|
|ATTENTION 3-32|1.03079242819832 |1.5593746412368 |3.39273691177368|
|ATTENTION 5-32|1.04141030765715 |1.5940105506352 |4.11833143234253|
|ATTENTION 3-16|1.07211020446959 |1.69643115316119|3.68811869621277|
|ATTENTION 5-64|1.05044556799389 |1.78613442466373|8.43711090087891|

#### Transfer Learning

| model                      | f1       | precision | recall   |
|----------------------------|----------|-----------|----------|
| ATTENTION 3-32, pretrained | 0.08709  | 0.04957   | 0.358123 |
| SAGE 3-32, pretrained      | 0.06452  | 0.03478   | 0.444412 |
| GCN  3-32, pretrained      | 0.061973 | 0.76422   | 0.395147 |

Lipo:
| model                    | mean absolute error | mean squared error | max error |
|--------------------------|---------------------|--------------------|-----------|
| ATTENTION 3-32, transfer | 1.01826             | 1.60076            | 5.21495   |
| GCN 3-32, transfer       | 0.97744             | 1.3891             | 3.44217   |
| SAGE 3-32, transfer      | 0.99493             | 1.46699            | 3.78525   |




#### Custom graph encodings

|model         |F1               |Recall          |Precision       |
|--------------|-----------------|----------------|----------------|
|ATTENTION 5-128, custom dataset|0.177496038034865|0.111776447105788|0.430769230769231|
|ATTENTION 5-64, custom dataset|0.138476755687438|0.0794551645856981|0.538461538461538|
|ATTENTION 5-32, custom dataset|0                |0               |0               |
|ATTENTION 5-16, custom dataset|0.135922330097087|0.0806916426512968|0.430769230769231|
|ATTENTION 3-128, custom dataset|0.123239436619718|0.0695825049701789|0.538461538461538|
|ATTENTION 3-64, custom dataset|0                |0               |0               |
|ATTENTION 3-32, custom dataset|0.115321252059308|0.0645756457564576|0.538461538461538|
|ATTENTION 3-16, custom dataset|0.120300751879699|0.0674789128397376|0.553846153846154|


|model         |mean absolute error|mean squared error|Max Error       |
|--------------|-------------------|------------------|----------------|
|GCN 3-64, custom dataset|0.912502304712931  |1.23040319368953  |3.11576461791992|
|GCN 5-64, custom dataset|0.931715891474769  |1.24643364804132  |3.13380861282349|
|GCN 5-128, custom dataset|0.929680191902887  |1.2685862722851   |3.63488841056824|
|GCN 3-128, custom dataset|0.93635904618672   |1.28365045757521  |3.04546570777893|
|GCN 3-32, custom dataset|0.932754458132244  |1.28878756051972  |3.30693531036377|
|GCN 5-16, custom dataset|0.944897433121999  |1.35001425288972  |3.69466733932495|
|GCN 5-32, custom dataset|0.95581499508449   |1.36479978022121  |4.01023101806641|
|GCN 3-16, custom dataset|0.963180348123442  |1.38878712305571  |4.3234124412    |

#### Graph level attributes

|model         |F1               |Recall          |Precision       |
|--------------|-----------------|----------------|----------------|
|ATTENTION 5-128, descriptors|0.150110375275938|0.0876288659793814|0.523076923076923|
|ATTENTION 5-64, descriptors|0.157835400225479|0.0924702774108322|0.538461538461538|
|ATTENTION 5-32, descriptors|0.156142365097589|0.0917678812415655|0.523076923076923|
|ATTENTION 5-16, descriptors|0.168978562421185|0.10105580693816|0.515384615384615|
|ATTENTION 3-128, descriptors|0.157164316829124|    0.098266128052391 |0.392312361492363|
|ATTENTION 3-64, descriptors|0.139489194499018|0.079954954954955|0.546153846153846|
|ATTENTION 3-32, descriptors|0.175742574257426|0.104719764011799|0.546153846153846|
|ATTENTION 3-16, descriptors|0.151515151515152|0.0892857142857143|0.5             |


|model         |mean absolute error|mean squared error|Max Error       |
|--------------|-------------------|------------------|----------------|
|GCN 5-64, descriptors|1.14554413273221   |11.1594213951202  |63.8460159301758|
|GCN 5-32, descriptors|1.14276210296722   |12.7610094399679  |69.0705108642578|
|GCN 5-16, descriptors|0.970055227052598  |1.36229835067477  |3.81348037719727|
|GCN 3-64, descriptors|1.01371599719638   |1.65436596529824  |9.82572746276856|
|GCN 3-32, descriptors|0.987400636218843  |1.405355153765    |3.88957548141479|
|GCN 3-16, descriptors|0.966446550687154  |1.34005836248398  |3.67364239692688|


### Improvements
Contrastive learning: https://arxiv.org/pdf/2109.01116
Learning multiple attributes at the same time: [1] Sharma, B., Chenthamarakshan, V., Dhurandhar, A. et al. Accurate clinical toxicity prediction using multi-task deep neural nets and contrastive molecular explanations. Sci Rep 13, 4908 (2023). https://doi.org/10.1038/s41598-023-31169-8

AUC LOSS: Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification, https://arxiv.org/abs/2012.03173

GENERALIZED AGGREGATION FUNCTION, novel graph normalization layers, :https://arxiv.org/pdf/2006.07739

ensembles and hypergraphs: https://github.com/zhangxwww/HyperFusion/blob/master/Multi_Model_Ensemble_on_Hypergraph.pdf
### Summary

# Building and running the application
Build and run the container
~~~
docker build . -t toxicml
docker run  -p 8081:8081 toxicml
~~~
