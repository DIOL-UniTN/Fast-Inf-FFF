# Fast-Inf: Ultra-Fast Embedded Intelligence on the Batteryless Edge

This is the repo associated to the paper `Fast-Inf: Ultra-Fast Embedded Intelligence on the Batteryless Edge`, to be published at ACM SenSys 2024.

To reproduce the experiments, please run:
```bash
python fff_experiment_mnist.py <leaf_width> <depth> <epochs> <norm_weight>
```
```bash
python fff_experiment_har.py <leaf_width> <depth> <epochs> <norm_weight>
```
```bash
python fff_experiment_speech_mfcc.py <leaf_width> <depth> <epochs> <norm_weight>
```
Note that, when norm_weight is set to `0`, the model is equivalent to a fast-feedforward network, while setting it to a value `> 0` makes the model will be trained with the L2 penalty as in the Fast-Inf paper. 


## To cite this code:

```bibtex
@inproceedings{custode2024fast,
  title={Fast-Inf: Ultra-Fast Embedded Intelligence on the Batteryless Edge},
  author={Custode, Leonardo Lucio and Farina, Pietro and Yıldız, Eren and Kılıç, Renan Beran and Yıldırım, Kasım Sinan and Iacca, Giovanni},
  year={2024},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 22th ACM Conference on Embedded Networked Sensor Systems},
}
```
