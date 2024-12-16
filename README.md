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

- Baseline Models: Logistic regression has the best precision (0.588235) but very low recall (0.153846). Naive Bayes achieves the highest recall (0.738462) but at the cost of very low precision (0.042572), leading to poor F1 scores. Random Forest balances precision and recall slightly better.
- Optimized Models: Random Forest optimization results in the best F1 score (0.275000), improving its balance between precision and recall. Naive Bayes optimized excels in recall but still suffers from poor precision, while logistic regression optimization underperforms overall.
- Conclusion: Random Forest performs best after optimization, but handling the class imbalance is key to further improving F1 scores.

| model                        | mean absolute error | mean squared error | max error    |
|------------------------------|---------------------|--------------------|--------------|
| Lasso                        | 1.006285            | 1.484697           | **3.261751** |
| Ridge                        | 0.683129            | 0.935416           | 9.890663     |
| Random Forrest               | 0.614670            | 0.629434           | 3.525267     |
| Lasso optimized              | 0.695592            | 0.910692           | 8.294026     |
| Ridge optimized              | 0.679734            | 0.926124           | 9.812714     |
| **Random Forrest optimized** | **0.609851**        | **0.619291**       | 3.591070     |

- Baseline Models: Random Forest outperforms Lasso and Ridge, achieving the lowest MSE (0.629434) and MAE (0.614670), showcasing its suitability for the small dataset.
- Optimized Models: Optimization improves Random Forest slightly, achieving the best results (MSE: 0.619291, MAE: 0.609851), while Lasso and Ridge see marginal gains but remain less effective.
- Conclusion: Random Forest is the best choice for this regression task, excelling in both baseline and optimized settings.




### Graph Neural Networks

Graph Neural Networks (GNNs) are a class of deep learning models designed to process data represented as graphs, making them particularly well-suited for molecular toxicity prediction. Molecules can naturally be represented as graphs, where atoms are nodes and chemical bonds are edges. GNNs excel at capturing the complex relationships between these components, leveraging both local (atom-level) and global (molecule-level) structural features. Unlike traditional methods that rely on predefined molecular descriptors, GNNs learn relevant features directly from raw molecular graphs, enabling more flexible and accurate modeling. However, GNNs can be computationally expensive, especially for large or highly complex molecular graphs, and may require extensive hyperparameter tuning and large amounts of labeled data to achieve optimal performance. Additionally, they may struggle with interpretability, as the learned features can be difficult to relate to traditional chemical intuition.

For starters, we will make use of the provided data loader from [Ogb](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py). The data loader converts the molecules into a set of atoms and defines a 9-dimensional feature vector for each atom. The connections are represented in an adjacency matrix and then together with the atom embeddings converted to a [PyG data object](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.data.Data.html#torch_geometric.data.Data)

#### Establishing a baseline
To establish a baseline, we will utilize a simple GCN architecture consisting of a graph convolutional layer, as described by Thomas [Thomas N. Kipf, Max Welling](https://arxiv.org/abs/1609.02907), followed by a ReLU activation function and a mean aggregation function to generate a single feature vector for the entire graph, which is then passed through a linear layer to produce the final output. 

Because the HIV dataset is unbalanced, we will also experiment with the weighted loss function and sampling based on the [Inbalanced Sampler](https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/loader.html#torch_geometric.loader.ImbalancedSampler) from PyG


| model                   | f1      | precision | recall  |
|-------------------------|---------|-----------|---------|
| base GNN                | 0       | 0         | 0       |
| base GNN, weighted loss | 0       | 0         | 0       |
| base GNN, sampling      | 0.05141 | 0.4415    | 0.03166 | 

The base GNN model shows no performance (all metrics are 0), indicating it fails to learn meaningful patterns from the unbalanced dataset without additional strategies. Using a weighted loss function does not improve performance, suggesting that this approach alone may not adequately address the severe class imbalance in the dataset. Incorporating sampling through the Imbalanced Sampler yields a slight improvement, with a modest F1 score (0.05941), precision (0.4815), and recall (0.03166). This suggests that sampling helps the model detect some positive samples but still struggles to balance precision and recall effectively. The baseline metrics are much lower than the Traditional ML methods, and the models will probably require a much bigger time investment than the traditional methods to achieve the same performance.

| model    | mean absolute error | mean squared error | max error |
|----------|---------------------|--------------------|-----------|
| base GNN | 1.048               | 1.63               | 3.408     |

The base GNN achieves a mean absolute error (1.048) and mean squared error (1.63), which are relatively high compared to traditional ML methods like Random Forest (MSE: 0.629434) in the earlier discussion.

#### Trying Different model architectures

In this section, we will focus on improving the baseline performance using bigger models and more advanced architectures. First, let's define a convolutional block as a sequence of:

1. Dropout operation
2. Convolutional operation
3. Activation operation
4. Batch normalization operation

We can then stack multiple convolution blocks to easily increase the depth of our model, and change the size of the atom embeddings to increase the size of each layer. Finally, the convolution blocks need to be followed by an aggregation operation and a multi-layer perceptron. We will also experiment with using different types of convolutions such as [SAGE Convolution](https://arxiv.org/abs/1706.02216) block and an [attention convolution block](https://arxiv.org/abs/2105.14491). 


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

Attention-based models exhibit slightly better recall compared to SAGE but suffer from lower precision. The best-performing attention-based model, ATTENTION 3-32, achieves an F1 score of 0.0724, with reasonable recall (0.0396). The embedding size and number of layers, don't seem to have a drastic effect on the performance.



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

The GCN models consistently outperform both SAGE and Attention models, achieving the lowest Mean Absolute Error (MAE) and Mean Squared Error (MSE). For example, GCN 3-32 achieves an MAE of 0.9836 and an MSE of 1.4827, the best among all models. SAGE models show a wide range of performance, with SAGE 5-64 achieving a competitive MAE (0.9965) but a high max error (7.5455), indicating poor handling of extreme values. Attention-based models generally perform worse, with higher MAE and MSE values compared to GCN and SAGE. For instance, ATTENTION 5-64 has the highest MSE (1.7861) and struggles with larger max errors (e.g., 8.4371).

We did manage to slightly improve our models performance, but compared to the traditional methods, the numbers are still low. 

#### Transfer Learning
Transfer learning involves pretraining a model on a large, related dataset and then fine-tuning it on a smaller target dataset. For the HIV and Lipo datasets, this could be done by retraining the model on one dataset, and fine-tuning it on the other. This could help us learn more robust features, especially when pretraining on the Hiv dataset as it is much larger than the Lipo dataset.

| model                      | f1       | recall    | precision|
|----------------------------|----------|-----------|----------|
| ATTENTION 3-32, pretrained | 0.08709  | 0.04957   | 0.358123 |
| SAGE 3-32, pretrained      | 0.06452  | 0.03478   | 0.444412 |
| GCN  3-32, pretrained      | 0.061973 | 0.76422   | 0.395147 |


| model                    | mean absolute error | mean squared error | max error |
|--------------------------|---------------------|--------------------|-----------|
| ATTENTION 3-32, transfer | 1.01826             | 1.60076            | 5.21495   |
| GCN 3-32, transfer       | 0.97744             | 1.3891             | 3.44217   |
| SAGE 3-32, transfer      | 0.99493             | 1.46699            | 3.78525   |


Overall we can see that transfer learning did have a small effect on the performance of our models, the performance could be greatly improved if we used an even larger dataset of molecules with similar tasks and trained them on molecule-level features, that can easily be computed using Rdkit. The downside is that pretraining on large molecular datasets can be resource-intensive, especially with advanced GNN architectures.

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
