# SRPO [Official Code]

### Super-Resolution by Predicting Offsets: An Ultra-Efficient Super-Resolution Network for Rasterized Images. [(Paper Link)]() 
- By [Jinjin Gu](https://scholar.google.com/citations?user=uMQ-G-QAAAAJ&hl=en&oi=ao), [Haoming Cai](https://scholar.google.com/citations?user=mePn76IAAAAJ&hl=en), Chenyu Dong, Ruofan Zhang, [Yulun Zhang](), [Wenming Yang](https://scholar.google.com/citations?hl=en&user=vsE4nKcAAAAJ), [Chun Yuan](https://scholar.google.com/citations?hl=en&user=fYdxi2sAAAAJ). In ECCV, 2022.

## Important Notes
- **Currently, We can only release well-trained SRPO pth file, part code of for inference without model, and code for blending due to commercial reason. However, with description in our paper, you can easily construct the concise SRPO and load our well-traiend pth.**
	1. SRPO_Blend produce the final output with offset-sr and offset, which are obtained from SRPO.
	1. SRPO_Inference lacks model detail. You should construct the SRPO by your own based on pth file or paper's description. Finally you could obtain offset-sr and offset from this file. I will add more detailed description later.
    1. SRPO_pth contains the well-trained SRPO on x2 scale. 

## Updates
- Last Update : 2022 Jul 20 by Haoming Cai