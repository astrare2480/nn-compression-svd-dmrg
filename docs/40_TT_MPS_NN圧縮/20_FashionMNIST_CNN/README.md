# Fashion-MNIST CNNのTT圧縮

小さい`Conv2d`によるTT-matrix表現、dense kernelを介さないdirect forwardとbackwardの整合性から始め、その実装を学習済みFashion-MNIST CNNの`conv2`へ接続して検証する。

## 結果・考察ノート

- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/00_TT-Conv表現とdense_kernel再構成]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/01_TT-Conv直接forwardとbackward等価性]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/02_学習済みCNNへのexact_TT-Conv接続]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/03_学習済みCNNのconv2_TT-rank_sweep]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/04_conv2_TT-cores-only_fine-tuning]]

## 対応Notebook

- [00_ttconv_representation_and_dense_kernel_reconstruction.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/00_ttconv_representation_and_dense_kernel_reconstruction.ipynb)
- [01_ttconv_direct_forward_and_backward_equivalence.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/01_ttconv_direct_forward_and_backward_equivalence.ipynb)
- [02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb)
- [03_pretrained_cnn_conv2_tt_rank_sweep.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/03_pretrained_cnn_conv2_tt_rank_sweep.ipynb)
- [04_pretrained_cnn_conv2_tt_cores_only_finetuning.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/04_pretrained_cnn_conv2_tt_cores_only_finetuning.ipynb)

## 技術ノート

- [[00_基礎理論/03_モデル圧縮理論/31_TTConv圧縮/79_Conv2dのTT-matrix表現と直接forward]]
