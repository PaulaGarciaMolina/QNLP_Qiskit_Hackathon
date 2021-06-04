# QNLP Qiskit Hackathon
Quantum Natural Language Processing (QNLP)'s project repository for Qiskit Hackathon Europe. Team members: Edwin Agnew, Pablo Díez-Valle, Paula García-Molina, María Hita-Pérez,  and Carlos Vega

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
### Semantic interpretation: ...
### Sentiment analysis: 
#### Statevector simulator:
- 1vs1: 63% (cross entropy), 77% (train error).
- 1vsAll: 75% (cross entropy).
- 2-qubit: 81 % (cross entropy).
#### Qasm simulator:
- 1vs1: 79% (cross entropy), 87% (train error).
#### ibmq_16_melbourne noisy simulator:
- 1vs1: 78% (cross entropy).

1vs1 results are for 2000 iterations of the SPSA optimizer, except for the noisy simulations (500 iterations) due to time requirements. For 1vsAll and 2-qubit multi-class classification we perform 1250 iterations.

For more details see our report here (link).

Packages used:
- qiskit 0.26.2
- discopy 0.3.5
- pytket 0.7.2
