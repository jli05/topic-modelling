# Topic Modelling Examples
We test the following algorithms for Topic Modelling

* NMF (Non-negative Matrix Factorisation)
* LDA Online Variational Inference
* Spectral LDA <https://github.com/Mega-DatA-Lab/SpectralLDA> 

We see that NMF is always a solid baseline in terms of speed, robustness, and result interpretability.

## Sample Results
We show results on subset of documents containing phrase "United States", within the corpus of Simple Wikipedia Page-Articles.


### NMF


```python
nmf = NMF(n_components=10, init='random', alpha=.3, l1_ratio=1e-6)
nmf.fit(tfidf)
print_top_words(nmf, tfidf_vectorizer.get_feature_names(), n_top_words=20)
```

    Topic #0: American Category actor New York ndash He television actors actress politician singer born footballer German writer English The British 2014
    Topic #1: County Alabama Area code Census Information Location List census counties City Zone Texas GNIS county area_land_sq_mi area_water_sq_mi area_total_sq_mi State elevation_ft
    Topic #2: movies movie Category drama comedy Movies American br The It set language released directed crime italictitle English thriller romantic United
    Topic #3: Illinois Cities US city geo Chicago stub States United Category Township map state Cook County Park Indiana City Democratic Mayor
    Topic #4: Kentucky Cities city geo US States stub United County Category seats Kenton Ohio Harlan Hills Hardin Indiana Louisville Oklahoma county
    Topic #5: States United Category Commonscat REDIRECT Party state Republican Arkansas President US establishments Democratic Senate Establishments New California Secretary America Washington
    Topic #6: Iowa Cities city geo States United stub US Category seats City Central 641 712 Moines 19 Des CDT CST Sioux
    Topic #7: ref http com www url title cite accessdate web The publisher date news html org https 2016 author work 2015
    Topic #8: br music album align The band Records rock song small center single groups released albums style com songs Billboard bands
    Topic #9: Florida Virginia towns Idaho US geo Cities city Arkansas States stub United Oklahoma Category Towns County Indiana town state Beach
    


### Latent Dirichlet Allocation

Variational Inference for LDA


```python
lda = LatentDirichletAllocation(n_components=10,
                                doc_topic_prior=1,
                                topic_word_prior=1,
                                max_iter=20,
                                learning_method='online')
lda.fit(tf)
print_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words=20)
```

    Topic #0: ref http www com title url cite accessdate web publisher date html news https 2011 2016 The work 2012 org
    Topic #1: County States United New city state Category nbsp City Area Jersey US River Florida Alabama State stub Census North Population
    Topic #2: Category States United He American Party nowrap Army New br University January York US jpg Republican Democratic President John DEFAULTSORT
    Topic #3: align center bgcolor style small br flagicon Grand left background Prix text rowspan Ret right width efcfff flag cfcfff colspan
    Topic #4: The ref In people It wikt This They called United year used time He book thumb States like jpg pages
    Topic #5: The Category br movie American music album movies series It film television com song Award Records United released TV Best
    Topic #6: sort WWE World team Category Championship League Canada Wrestling Cup The football match year hockey br wrestling Team dash Olympics
    Topic #7: font User talk color UTC span style sup article The Wikipedia page nbsp small Special articles think 2009 user face
    Topic #8: The File nbsp flag br code United nowiki svg link jpg word It language replace type country convert Typo used
    Topic #9: American ndash The born actor dies British actress German English singer politician French player United writer John President FA BeenOnMainPage
    


### Spectral LDA


```python
import sys
sys.path.insert(0, '../SpectralLDA')

from spectral_lda import spectral_lda
```

    Using numpy backend.
    Using numpy backend.



```python
alpha, beta = spectral_lda(tf, alpha0=10, k=10)
print_top_words_factors(beta.T, tf_vectorizer.get_feature_names(), 20)
```

    # docs: 30285	# valid: 30285


    Topic #0: County Category Kentucky geo US stub Cities United city States Florida Idaho towns Oklahoma Illinois seats county state Texas Arkansas
    Topic #1: Category States United Commonscat American REDIRECT state New California Settlements movies York movie establishments deaths television century Geography nationality Buildings
    Topic #2: States American movies movie New television York California Settlements He state series deaths It Movies actors seats drama DEFAULTSORT br
    Topic #3: Iowa States United Cities Category geo stub city US Illinois Kentucky towns Idaho Florida Arkansas Indiana Oklahoma seats Commonscat state
    Topic #4: stub ref http www The br com title Commonscat url cite accessdate web publisher American date align He New html
    Topic #5: city ref The http www com br title url cite date accessdate web In It He publisher American music small
    Topic #6: geo ref The http www com br title url American cite date He accessdate web It publisher Infobox jpg References
    Topic #7: Category REDIRECT ref The http redirect br www title com url cite accessdate web In 2010 publisher small align code
    Topic #8: Virginia geo towns stub States United Category Towns town US Idaho Florida Arkansas Cities state Indiana Oklahoma Illinois West Commonscat
    Topic #9: States United REDIRECT Party President Commonscat Rights Bill units Census History dollar Constitution Republican presidential Senate redirect Department Army Secretary
    
