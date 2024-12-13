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

If any of the steps fails the whole pipeline fails
## Modeling
### Dataset description
### Traditional ML methods
#### Performance and evaluation
### Graph Neural Networks
#### Graph Encoding
#### Overview of Graph Methods
#### Establishing a baseline
#### Trying Different model architectures
#### Transfer Learning
#### Custom graph encodings
### Improvements
### Summary

# Building and running the application
Build and run the container
~~~
docker build . -t toxicml
docker run  -p 8081:8081 toxicml
~~~
