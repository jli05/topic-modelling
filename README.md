# Topic Modelling Experiments
We test the following algorithms for Topic Modelling

* NMF (Non-negative Matrix Factorisation)
* LDA Online Variational Inference
* Spectral LDA <https://github.com/Mega-DatA-Lab/SpectralLDA-Spark>

# Training Details
Trained on the [UCI Bag of Words NYTimes Dataset](https://archive.ics.uci.edu/ml/datasets/bag+of+words) and [Simplewiki Pages-Articles Dump](https://dumps.wikimedia.org/simplewiki/latest/). AWS single node with 8 vCPU + 16G Memory.

1. NMF: the L2-penalty coefficient = 0.3
2. LDA Online Variational: the Dirichlet prior distribution with all-one vector for the Topic distribution or Doc-Word distribution to allow non-informative prior. The batch size is adjusted to fully feed all cores. 10 full epochs on the entire corpus.
3. Spectral LDA: `alpha_0 = numTopics` to allow non-informative prior. The remaining parameters are as default.

# Findings
Note that NMF treats Topic Modelling as a single matrix decomposition problem and falls in the P class. The LDA formulation falls in the NP-hard class. The Spectral LDA, by mildly relaxing certain constraints, essentially formulates a two-level (hierarchical) decomposition problem and solves with purely linear algebra. The LDA Online Variational Inference follows Hoffman and Blei.  

1. NMF trains fast but it is very sensitive to extremely high IDF (rare) words, if there're too many in the corpus, all the topics will be filled with these rare words and exhibit no consistency or interpretability for any topic. When it's a small corpus and there aren't many rare words, NMF's result could be good.

2. Spectral LDA is CPU-efficient but is sensitive to network communication cost. When the topics has intrinsically as many self-coherent topics as computed for, the topics discovered by Spectral LDA appear particularly pertinent and self-coherent.

3. LDA Online Variational generally produce good and self-coherent topics, but some topics could be too "general" even for given collection of documents around a known central theme. 

