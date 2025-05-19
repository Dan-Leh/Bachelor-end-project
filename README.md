# Bachelor end project 2021 Dan Lehman

This is the repository for my 2022 Bachelor End Project at TU/e.

Abstract—This paper concerns a comparison between adversarial discriminative learning and teacher-student learning to perform unsupervised domain adaptation from daytime to nighttime images. The two different learning approaches were implemented in training on a so-called multi-head network consisting of a shared encoder that learns features for daytime and nighttime images alike and two separate decoders, one for daytime images and one for nighttime images. We found that teacher-student domain adaptation led to an improvement in performance, especially on nighttime images, compared to adver- sarial discriminative domain adaptation. Additionally, we tested various ensemble methods to merge the outputs of our multi-head network’s decoders and found them to not have a major effect on the network’s performance. Lastly, we show that a teacher-student trained single head network that predicts segmentation masks for both day- and nighttime images performed nearly as well as the multi-head network while being simpler to implement and smaller in size.

[Click here](https://drive.google.com/file/d/19MhpWJcPvhns5TTrIxLrtpIsIR7TylM3/view?usp=sharing) for the bachelor thesis detailing my findings.

The contents of the paper and the code are a continuation of the work commenced by a previous [master thesis](https://drive.google.com/drive/folders/1vLEvAW_X31_8gtIPl380kn8PTQhHSmin?usp=sharing).

The folder "Ensembles" contains
- "Smallclassifier.ipynb", the code used to train the day/night classifier
- "Evaluation_ensembles", which evaluates the test set using by the classifier, soft voting and maximum likelihood.

The folder "Domain adaptation models" contains
- "MH-TSDA", the Multi-Head network with Teacher-Student Domain Adaptation
- "SH-TSDA", the Single Head network with Teacher-Student Domain Adaptation
