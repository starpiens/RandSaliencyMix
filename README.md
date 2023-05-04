# RandSaliencyMix
SaliencyMix [[1]](#ref1) with more randomness. 

## Introduction
Recently the size of the vision models has been rapidly increasing, resulting powerful representation ability of the model. However, the dataset size needed to train such a model without overfitting is also growing fast. But obtaining more data is often difficult, limiting the practical applicability of the large vision models. Data augmentation is one of the most efficient strategies to alleviate the issue. CutMix [[2]](#ref2) randomly replaces a random image region with a random patch from another training image. SaliencyMix [[1]](#ref1) further improves this idea using a saliency map, assuring selected random patch contains relevant information about the source object. However, SaliencyMix [[1]](#ref1) fixes the location of the patch being cropped and the location of the region being replaced. We think its deterministic “crop-and-paste” strategy limits the full potential of SaliencyMix [[1]](#ref1) because of the lack of diversity of training data. Therefore, we aim to find a better “pasting” strategy that cares about the region being replaced. We will compare our strategy with the random pasting strategy of [[1]](#ref1) and [[2]](#ref2).

## Goal
Improving the data augmentation technique of the target paper [[1]](#ref1) by adding more randomized behavior. 

## Plan
- Implement random location cropping/pasting proportional to saliency intensity. 
- Implement a label mixing algorithm considering not only patch size but also saliency intensity. 
- Implement a hybrid method of CutMix [[2]](#ref2) and MixUp [[3]](#ref3), which can further increase the diversity of the training dataset.

## References

<p id="ref1"><b>[1]</b> A F M Shahab Uddin and Mst. Sirazam Monira and Wheemyung Shin and TaeChoong Chung and Sung-Ho Bae. “SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization.” <i>International Conference on Learning Representations.</i> 2021.</p>

<p id="ref2"><b>[2]</b> Sangdoo Yun and Dongyoon Han and Seong Joon Oh and Sanghyuk Chun and Junsuk Choe and Youngjoon Yoo. “CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.” <i>Proceedings of the IEEE/CVF International Conference on Computer Vision.</i> 2019.</p>

<p id="ref3"><b>[3]</b> Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz. “Mixup: Beyond Empirical Risk Minimization.” <i>International Conference on Learning Representations.</i> 2018.</p>
