# QNLP Qiskit Hackathon
Quantum Natural Language Processing (QNLP)'s project repository for Qiskit Hackathon Europe. Team members: Edwin Agnew, Pablo Díez-Valle, María Hita, Paula García-Molina, and Carlos Vega

## Outline
This repository is split into two directories:
- *semantic_interpretation*: notebooks and datasets for determining similarity between English and Spanish sentences
- *sentiment_analysis*: notebooks and datasets for classifying sentences into one of your sentiments: happy, sad, angry, scared 

## Methodology
Our methodology is largely based off (Lorenz et al.) and (Other) and uses the following structure:
- Create a dataset of sentences and split into training and testing sets
- Convert sentences into DisCoPy sentence diagrams
- Convert sentence diagrams into parameterized quantum circuits using the IQP ansatz
- Optimize parameters by applying SPSA variational algorithm on training sentences

## Results
We obtained the following results:
- Semantic interpretation: ...
- Sentiment analysis: 70% accuracy using a multi-class classifier and 90% accuracy using 6 1 versus 1 classifiers.

For more details see our report here (link).

Packages used:
- qiskit 0.26.2
- discopy 0.3.5
- pytket 0.7.2
