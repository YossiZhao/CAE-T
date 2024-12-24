# CwA-T: A Channelwise AutoEncoder with Transformer for EEG Abnormality Detection

This is the repository for ["CwA-T: A Channelwise Autoencoder with Transformer for EEG Abnormality Detection"](https://github.com/YossiZhao/CAE-T). You can checkout the [PDF file](https://github.com/YossiZhao/CwA-T/paper/CwA-T.pdf) of our paper in our GitHub repo.

- If you have used our code or referred to our results in your research, please cite:

```bibtex
@misc{zhao2024caetchannelwiseautoencodertransformer,
      title={CwA-T: A Channelwise Autoencoder with Transformer for EEG Abnormality Detection}, 
      author={Youshen Zhao and Keiji Iramina},
      year={2024},
      eprint={2412.14522},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.14522}, 
}
```
---

## Introduction

The proposed **CwA-T framework** is a novel method for EEG abnormality detection, combining a channelwise CNN-based autoencoder with a single-head transformer classifier. This approach effectively addresses challenges in EEG analysis, such as the high dimensionality, complexity of long-term signals, and the need for biologically interpretable features.

Key highlights of our study:
- **Efficient Processing**: Handles long-term EEG signals with reduced computational costs.
- **Interpretability**: Retains biologically meaningful features, aligning with EEG channel independence.
- **Competitive Results**: Achieved 85.0% accuracy, 76.2% sensitivity, and 91.2% specificity on the TUH Abnormal EEG Corpus.

The framework’s workflow involves:
1. Preprocessing raw EEG signals.
2. Feature extraction using the channelwise autoencoder.
3. Classification using the transformer model.

### Workflow Overview

Below is a visual representation of the CwA-T workflow:

![Workflow Overview](https://github.com/YossiZhao/CAE-T/blob/v1.2/images/Overview.jpeg)

---
