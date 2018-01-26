# Reef: Overcoming the Barrier to Labeling Training Data
*This is the first version of the README.*

Reef is an automated system for labeling training data based on a small labeled dataset. Reef utilizes ideas from program synthesis to *automatically* generate a set of interpretable heuristics that are then used to label unlabeled training data efficiently. 

## Installation
**TODO: Any help appreciated!**
Currently independent of other repos (like Snorkel. If you have Snorkel, can set a flag [here]( ) as True). There is one semi-supervised method in `experimental_testing.ipynb`  that requires some optimization library but it’s not via Python. Therefore ignore it :D

## Reef Workflow Overview
The inputs to Reef are the following: 
* A labeled dataset, which contains a numerical feature matrix and a vector of ground truth labels (currently only supports binary classification)
* An unlabeled dataset, which contains a numerical feature matrix

 The following is the overall workflow Reef follows to label training data automatically. The overall process is encoded in `tutorial.ipynb` and the main file `program_synthesis/heuristic_generator.py`
1. Using the labeled dataset, Reef generates heuristics like decision trees, or small logistic regression models.  The synthesis code is in `program_synthesis/synthesizer.py`. 
	1. A heuristic is generated for each possible combination of `c` features, where `c` is the cardinality. For example, with `c=1` and 10 features, 10 heuristics will be generated.
	2. For each generated heuristic, a [*beta* parameter](https://github.com/HazyResearch/reef/blob/4bb29e26ec99c4ab99d0cb644183ff2df35abfa9/program_synthesis/synthesizer.py#L94) is calculated. This represents the minimum confidence level at which the heuristics will assign a label. This is done by maximizing the F1 score on the labeled dataset. 
2. These heuristics are passed to a [pruner](https://github.com/HazyResearch/reef/blob/4bb29e26ec99c4ab99d0cb644183ff2df35abfa9/program_synthesis/heuristic_generator.py#L51) that selects the best heuristic by maximizing a combination of the F1 score on the labeled dataset and diversity in terms of how many points it labels that previously selected heuristics don’t. 
3. The selected heuristic and previously chosen heuristics are then passed to the [verifier](https://github.com/HazyResearch/reef/blob/4bb29e26ec99c4ab99d0cb644183ff2df35abfa9/program_synthesis/verifier.py#L21) which learns accuracies for the heuristics based on the labels the heuristics assign to the *unlabeled* dataset. 
4. Finally, Reef calculates the probabilistic labels the heuristics assign to the labeled dataset and pass datapoint with [low confidence labels](https://github.com/HazyResearch/reef/blob/4bb29e26ec99c4ab99d0cb644183ff2df35abfa9/program_synthesis/heuristic_generator.py#L141) to the synthesizer. We repeat this procedure in an iterative manner. 

## Tutorial
**TODO: find nice dataset for tutorial**
The tutorial is is based on a dataset used for anomaly detection. We provide a way to visualize the heuristics generated and the features they are based on. We cannot compare to baseline methods since we do not have a held out test set yet, but hopefully the fact that validation accuracy is not 100% convinces you about the not overfitting claim :)
