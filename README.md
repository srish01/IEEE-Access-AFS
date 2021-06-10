# IEEE-Access---Gradient-Based-Attacks-againstInformation-Theoretic-Feature-Selection

Work in this repo has been submitted to IEEE Access

### Abstract

Machine learning (ML) is vital to many application-driven fields, such as signal classification, cyber-security, and image classification. Unfortunately, many of these fields can easily have their training data tampered with by an adversary to thwart an ML algorithm's objective. Further, the adversary can affect any stage in an ML pipeline (e.g.,  learning, preprocessing, and classification). Recent work has shown that many models can be attacked by poisoning the training data, and the impact of the poisoned data can be quite significant. Prior works on adversarial feature selection have shown that the attacks can damage feature selection process. Information-theoretic feature selection algorithms are widely used for their ability to model nonlinear relationships, classifier independence and lower computational requirements. One important question from the security perspective of these widely used approaches is, whether filter-based algorithms are robust against other feature selection attacks. In this work, we focus on the task of information-theoretic feature selection and the impact that an adversary can have on these selections. We study the transferability of feature selection attacks not designed for information-theoretic approaches. Since, many existing gradient-based attacks are designed for LASSO, we study the transferability of attacks from LASSO to information-theoretic feature selection algorithms like MIM, mRMR, JMI, etc. We show that the stability of information-theoretic feature selection can be significantly degraded by injecting poisonous data into the training dataset.


### Folder Description
#### Excel_file: 
This folder contains spreadsheet for each dataset holding all the results

#### results:
results folder contain the .npz files of results including : Jaccard distance, Kuncheva distance, Jaccard Consistency, Kuncheva Consistency, and poisonind samples

#### line-graphs
line graphs folder contains the graphs depicting trends and patterns from results obtained

#### data
data folder contains bunch of UCI datasets, from which 5 datasets were used in this work:
1. conne-bench-sonar-mine-rocks
2. molec-biol-promoter
3. Breast-cancer-wisc-prog
4. ionosphere
5. twonorm

This folder has clean and adversarial data
