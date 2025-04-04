# HwaMei-500

HwaMei-500 is a medical information extraction corpus consisting of 500 Chinese electronic medical records (EMRs). It provides three tasks, i.e., medical entity recognition (MER), medical relation extraction (MRE), and medical attribute extraction (MAE). Check our [published article](https://doi.org/10.1016/j.artmed.2023.102573) (Zhu et al. 2023) and the [annotation scheme](https://github.com/syuoni/eznlp/tree/master/publications/framework/scheme.pdf) for more details.

The released data have been manually *de-identified*. Specifically, we have carefully replaced the protected health information (PHI) mentions by realistic *surrogates* (Stubbs et al., 2015). For example, all the person names are replaced by combinations of randomly sampled family and given names, where the sampling accords to the frequencies reported by National Bureau of Statistics of China. All the locations are replaced by randomly sampled addresses in China. (In other words, all the PHI mentions are *fake* in the released data.) Such process preserves the usability of our data and prevent PHI leak simultaneously.

HwaMei-500 is available upon request. Please visit this [link](http://47.99.121.158:8000), sign and upload the data use agreement. Please strictly abide by the terms of the agreement. Contact liuyiyang@ucas.ac.cn if you need help.


## References
* Zhu, E., Sheng, Q., Yang, H., Liu, Y., Cai, T., and Li, J. A unified framework of medical information annotation and extraction for Chinese clinical text. *Artificial Intelligence in Medicine*, 2023, 142:102573.
* Stubbs, A., Uzuner, Ö., Kotfila, C., Goldstein, I., and Szolovits, P. Challenges in synthesizing surrogate PHI in narrative EMRs. *Medical Data Privacy Handbook*, 2015, 717–735.
