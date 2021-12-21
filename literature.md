# Literature for protein representation learning, generation and some misc. ideas

Guidelines: mark the papers already fully-read, keep chronological order (from earliest to latest). Ideally, add model acronyms when the title does not state them.


## General NLP

* Radford / Improving Language Understanding by Generative Pre-Training (GPT) / 2018 / **read**
* Devlin / BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding / 2019 / **read**
* Liu / RoBERTa: A Robustly Optimized BERT Pretraining Approach / 2019 / **read**
* Jain / Attention is not Explanation / 2019
* Wiegreffe / Attention is not not Explanation / 2019
* Xiong / On Layer Normalization in the Transformer Architecture / 2020
* Tay / Efficient Transformers: A Survey / 2020


## General SSL

* Kaku / Intermediate Layers Matter in Momentum Contrastive Self Supervised Learning / 2021


## General reviews and discussions

* Alley / Unified rational protein engineering with sequence-based deep representation learning (UniRep) / 2019
* Hoarfrost / Shedding Light on Microbial Dark Matter with A Universal Language of Life / 2020 / **read**
* Ofer / The language of proteins: NLP, machine learning & protein sequences / 2021
* Bepler / Learning the protein language: Evolution, structure, and function / 2021
* Detlefsen / What is a meaningful representation of protein sequences? / 2021
* Biswas / Low-N protein engineering with data-efficient deep learning / 2021 / **read**


## Protein and Biological Sequence Modeling

*This part should be further split into subsections.*

* Suzek / UniRef: comprehensive and non-redundant UniProt reference clusters / 2007
* Rao / Evaluating Protein Transfer Learning with TAPE / 2019 / **read**
* Vig / BERTology Meets Biology: Interpreting Attention in Protein Language Models / 2020
* Choromanski / Masked Language Modeling for Proteins via Linearly Scalable Long-Context Transformers / 2020
* Filipavicius / Pre-training Protein Language Models with Label-Agnostic Binding Pairs Enhances Performance in Downstream Tasks / 2020 / **read**
* Kim / Deep protein-ligand binding prediction using unsupervised learned representations / 2020 / **read**
* Amimeur / Designing Feature-Controlled Humanoid Antibody Discovery Libraries Using Generative Adversarial Networks / 2020
* Grechishnikova / Transformer neural network for protein‐specific de novo drug generation as a machine translation problem / 2021
* Rao / MSA Transformer / 2021
* Thumuluri / NetSolP: predicting protein solubility in E. coli using language models / 2021
* Elnaggar / ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Learning / 2021
* Rao / Transformer protein language models are unsupervised structure learners / 2021 / **read**
* Rives / Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences / 2021 / **read**
* Abanades / ABlooper: Fast accurate antibody CDR loop structure prediction with accuracy estimation / 2021
* Weinstein / Optimal Design of Stochastic DNA Synthesis Protocols based on Generative Sequence Models / 2021
* Townshend / Geometric deep learning of RNA structure / 2021


## Misc. ideas

Modified cross-entropy loss for supervised training with imbalanced data. Hard/easy example mining.

* Lin / Focal Loss for Dense Object Detection / 2017
* Xuan / Hard negative examples are hard, but useful / 2020
* Xuan / Improved Embeddings with Easy Positive Triplet Mining / 2020

Train (cross-modal) inference models on e.g. aligned sequences and labels, train unconditional generative models on sequences, then do latent space search of the generative representation in the "CLIP-guided style".

* Radford / Learning Transferable Visual Models From Natural Language Supervision (CLIP) / 2021
* Galatolo / Generating images from caption and vice versa via CLIP-Guided Generative Latent Space Search / 2021
* Kim1 / DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation / 2021

Or alternatively, train controlable generative model on top of unconditional generative latent space.

* Winter / Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations / 2019 && Auto-Encoding Molecular Conformations / 2021
* Méndez-Lucio / Cell morphology-guided de novo hit design by conditioning generative adversarial networks on phenotypic image features / 2020 / **read**
* Mittal / Symbolic Music Generation with Diffusion Models / 2021






