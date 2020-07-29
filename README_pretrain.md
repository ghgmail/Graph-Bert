# 只用预训练任务1，速度变快，效果降低
这也侧面证明了我们的预训练是有效果的
## 150轮预训练 cora
```python
python script_5_fine_tuning_cora.py
************ Start ************
GrapBert, dataset: cora, residual: graph_raw, k: 7, hidden dimension: 32, hidden layer: 2, attention head: 2
from_pretrain
<class 'code.MethodGraphBert.MethodGraphBert'> <class 'type'>
Loading cora dataset...
Load WL Dictionary
Load Hop Distance Dictionary
Load Subgraph Batches
Epoch: 0001 loss_train: 1.9190 acc_train: 0.2000 loss_val: 1.8130 acc_val: 0.3400 loss_test: 1.7758 acc_test: 0.3600 time: 0.1289s
Epoch: 0011 loss_train: 0.1492 acc_train: 1.0000 loss_val: 0.9522 acc_val: 0.7067 loss_test: 0.7906 acc_test: 0.7820 time: 0.1426s
Epoch: 0021 loss_train: 0.0065 acc_train: 1.0000 loss_val: 0.8633 acc_val: 0.7267 loss_test: 0.7484 acc_test: 0.7900 time: 0.1875s
Epoch: 0031 loss_train: 0.0023 acc_train: 1.0000 loss_val: 0.8724 acc_val: 0.7300 loss_test: 0.7639 acc_test: 0.7870 time: 0.1158s
Epoch: 0041 loss_train: 0.0037 acc_train: 1.0000 loss_val: 0.8219 acc_val: 0.7433 loss_test: 0.7189 acc_test: 0.7860 time: 0.1174s
Epoch: 0051 loss_train: 0.0073 acc_train: 1.0000 loss_val: 0.7799 acc_val: 0.7600 loss_test: 0.6790 acc_test: 0.7920 time: 0.1773s
Epoch: 0061 loss_train: 0.0098 acc_train: 1.0000 loss_val: 0.7794 acc_val: 0.7367 loss_test: 0.6791 acc_test: 0.7850 time: 0.1202s
Epoch: 0071 loss_train: 0.0091 acc_train: 1.0000 loss_val: 0.7821 acc_val: 0.7400 loss_test: 0.6830 acc_test: 0.7790 time: 0.1373s
Epoch: 0081 loss_train: 0.0081 acc_train: 1.0000 loss_val: 0.7858 acc_val: 0.7400 loss_test: 0.6906 acc_test: 0.7740 time: 0.1234s
Epoch: 0091 loss_train: 0.0081 acc_train: 1.0000 loss_val: 0.7806 acc_val: 0.7500 loss_test: 0.6901 acc_test: 0.7760 time: 0.1711s
Epoch: 0101 loss_train: 0.0077 acc_train: 1.0000 loss_val: 0.7843 acc_val: 0.7467 loss_test: 0.6938 acc_test: 0.7710 time: 0.1475s
Epoch: 0111 loss_train: 0.0071 acc_train: 1.0000 loss_val: 0.7877 acc_val: 0.7333 loss_test: 0.7014 acc_test: 0.7720 time: 0.1700s
Epoch: 0121 loss_train: 0.0064 acc_train: 1.0000 loss_val: 0.7938 acc_val: 0.7467 loss_test: 0.7051 acc_test: 0.7730 time: 0.1224s
Epoch: 0131 loss_train: 0.0059 acc_train: 1.0000 loss_val: 0.7916 acc_val: 0.7433 loss_test: 0.7073 acc_test: 0.7740 time: 0.1177s
Epoch: 0141 loss_train: 0.0061 acc_train: 1.0000 loss_val: 0.7980 acc_val: 0.7367 loss_test: 0.7124 acc_test: 0.7760 time: 0.1209s
Optimization Finished!
Total time elapsed: 21.5586s, best testing performance  0.793000, minimun loss  0.676881
************ Finish ************
```
## 1000轮预训练cora
```python
(py3) DESKTOP-901DE93 :: zk_file/project/Graph-Bert ‹master*› » python script_5_fine_tuning_cora.py
************ Start ************
GrapBert, dataset: cora, residual: graph_raw, k: 7, hidden dimension: 32, hidden layer: 2, attention head: 2
from_pretrain
<class 'code.MethodGraphBert.MethodGraphBert'> <class 'type'>
Loading cora dataset...
Load WL Dictionary
Load Hop Distance Dictionary
Load Subgraph Batches
Epoch: 0001 loss_train: 1.9275 acc_train: 0.1500 loss_val: 1.8350 acc_val: 0.3467 loss_test: 1.7907 acc_test: 0.4040 time: 0.1302s
Epoch: 0011 loss_train: 0.1411 acc_train: 1.0000 loss_val: 0.9300 acc_val: 0.7233 loss_test: 0.7655 acc_test: 0.7850 time: 0.1139s
Epoch: 0021 loss_train: 0.0061 acc_train: 1.0000 loss_val: 0.8319 acc_val: 0.7467 loss_test: 0.7021 acc_test: 0.7950 time: 0.1702s
Epoch: 0031 loss_train: 0.0023 acc_train: 1.0000 loss_val: 0.8451 acc_val: 0.7400 loss_test: 0.7157 acc_test: 0.7920 time: 0.1830s
Epoch: 0041 loss_train: 0.0039 acc_train: 1.0000 loss_val: 0.8002 acc_val: 0.7433 loss_test: 0.6842 acc_test: 0.7860 time: 0.1210s
Epoch: 0051 loss_train: 0.0080 acc_train: 1.0000 loss_val: 0.7767 acc_val: 0.7267 loss_test: 0.6655 acc_test: 0.7800 time: 0.1177s
Epoch: 0061 loss_train: 0.0103 acc_train: 1.0000 loss_val: 0.7820 acc_val: 0.7267 loss_test: 0.6753 acc_test: 0.7850 time: 0.1726s
Epoch: 0071 loss_train: 0.0089 acc_train: 1.0000 loss_val: 0.7899 acc_val: 0.7333 loss_test: 0.6815 acc_test: 0.7830 time: 0.1204s
Epoch: 0081 loss_train: 0.0086 acc_train: 1.0000 loss_val: 0.7869 acc_val: 0.7233 loss_test: 0.6830 acc_test: 0.7810 time: 0.1679s
Epoch: 0091 loss_train: 0.0084 acc_train: 1.0000 loss_val: 0.7882 acc_val: 0.7133 loss_test: 0.6842 acc_test: 0.7840 time: 0.1737s
Epoch: 0101 loss_train: 0.0080 acc_train: 1.0000 loss_val: 0.7911 acc_val: 0.6933 loss_test: 0.6865 acc_test: 0.7790 time: 0.1179s
Epoch: 0111 loss_train: 0.0075 acc_train: 1.0000 loss_val: 0.7911 acc_val: 0.6933 loss_test: 0.6890 acc_test: 0.7820 time: 0.1857s
Epoch: 0121 loss_train: 0.0070 acc_train: 1.0000 loss_val: 0.7984 acc_val: 0.6967 loss_test: 0.6889 acc_test: 0.7820 time: 0.1684s
Epoch: 0131 loss_train: 0.0066 acc_train: 1.0000 loss_val: 0.7976 acc_val: 0.7000 loss_test: 0.6978 acc_test: 0.7820 time: 0.1203s
Epoch: 0141 loss_train: 0.0062 acc_train: 1.0000 loss_val: 0.7978 acc_val: 0.7000 loss_test: 0.6952 acc_test: 0.7840 time: 0.1723s
Epoch: 0151 loss_train: 0.0062 acc_train: 1.0000 loss_val: 0.8024 acc_val: 0.6933 loss_test: 0.6959 acc_test: 0.7860 time: 0.0933s
Epoch: 0161 loss_train: 0.0057 acc_train: 1.0000 loss_val: 0.8094 acc_val: 0.7000 loss_test: 0.7076 acc_test: 0.7810 time: 0.1297s
Epoch: 0171 loss_train: 0.0053 acc_train: 1.0000 loss_val: 0.8099 acc_val: 0.6967 loss_test: 0.7017 acc_test: 0.7820 time: 0.1200s
Epoch: 0181 loss_train: 0.0053 acc_train: 1.0000 loss_val: 0.8111 acc_val: 0.7000 loss_test: 0.7058 acc_test: 0.7830 time: 0.1485s
Epoch: 0191 loss_train: 0.0052 acc_train: 1.0000 loss_val: 0.8151 acc_val: 0.7033 loss_test: 0.7050 acc_test: 0.7830 time: 0.1210s
Epoch: 0201 loss_train: 0.0051 acc_train: 1.0000 loss_val: 0.8188 acc_val: 0.6967 loss_test: 0.7167 acc_test: 0.7790 time: 0.1217s
Epoch: 0211 loss_train: 0.0051 acc_train: 1.0000 loss_val: 0.8212 acc_val: 0.6967 loss_test: 0.7179 acc_test: 0.7810 time: 0.1680s
Epoch: 0221 loss_train: 0.0047 acc_train: 1.0000 loss_val: 0.8191 acc_val: 0.7100 loss_test: 0.7038 acc_test: 0.7860 time: 0.1209s
Epoch: 0231 loss_train: 0.0047 acc_train: 1.0000 loss_val: 0.8262 acc_val: 0.7100 loss_test: 0.7227 acc_test: 0.7860 time: 0.1023s
Epoch: 0241 loss_train: 0.0045 acc_train: 1.0000 loss_val: 0.8228 acc_val: 0.6933 loss_test: 0.7190 acc_test: 0.7810 time: 0.1666s
Epoch: 0251 loss_train: 0.0050 acc_train: 1.0000 loss_val: 0.8397 acc_val: 0.7000 loss_test: 0.7282 acc_test: 0.7740 time: 0.1216s
Epoch: 0261 loss_train: 0.0039 acc_train: 1.0000 loss_val: 0.8287 acc_val: 0.7033 loss_test: 0.7184 acc_test: 0.7810 time: 0.1197s
Epoch: 0271 loss_train: 0.0048 acc_train: 1.0000 loss_val: 0.8303 acc_val: 0.6967 loss_test: 0.7240 acc_test: 0.7810 time: 0.1579s
Epoch: 0281 loss_train: 0.0040 acc_train: 1.0000 loss_val: 0.8321 acc_val: 0.6967 loss_test: 0.7313 acc_test: 0.7820 time: 0.1251s
Epoch: 0291 loss_train: 0.0042 acc_train: 1.0000 loss_val: 0.8425 acc_val: 0.7033 loss_test: 0.7310 acc_test: 0.7730 time: 0.1675s
Epoch: 0301 loss_train: 0.0045 acc_train: 1.0000 loss_val: 0.8391 acc_val: 0.6967 loss_test: 0.7284 acc_test: 0.7850 time: 0.1205s
Epoch: 0311 loss_train: 0.0039 acc_train: 1.0000 loss_val: 0.8486 acc_val: 0.7100 loss_test: 0.7378 acc_test: 0.7780 time: 0.1728s
Epoch: 0321 loss_train: 0.0043 acc_train: 1.0000 loss_val: 0.8328 acc_val: 0.7033 loss_test: 0.7237 acc_test: 0.7900 time: 0.1768s
Epoch: 0331 loss_train: 0.0039 acc_train: 1.0000 loss_val: 0.8535 acc_val: 0.7133 loss_test: 0.7437 acc_test: 0.7790 time: 0.1425s
Epoch: 0341 loss_train: 0.0038 acc_train: 1.0000 loss_val: 0.8569 acc_val: 0.6933 loss_test: 0.7373 acc_test: 0.7740 time: 0.1228s
Epoch: 0351 loss_train: 0.0039 acc_train: 1.0000 loss_val: 0.8545 acc_val: 0.7000 loss_test: 0.7471 acc_test: 0.7730 time: 0.1000s
Epoch: 0361 loss_train: 0.0043 acc_train: 1.0000 loss_val: 0.8429 acc_val: 0.7100 loss_test: 0.7336 acc_test: 0.7800 time: 0.1713s
Epoch: 0371 loss_train: 0.0034 acc_train: 1.0000 loss_val: 0.8502 acc_val: 0.7100 loss_test: 0.7457 acc_test: 0.7780 time: 0.1169s
Epoch: 0381 loss_train: 0.0039 acc_train: 1.0000 loss_val: 0.8457 acc_val: 0.7000 loss_test: 0.7375 acc_test: 0.7850 time: 0.1716s
Epoch: 0391 loss_train: 0.0036 acc_train: 1.0000 loss_val: 0.8648 acc_val: 0.6933 loss_test: 0.7428 acc_test: 0.7720 time: 0.1613s
Epoch: 0401 loss_train: 0.0040 acc_train: 1.0000 loss_val: 0.8514 acc_val: 0.6967 loss_test: 0.7589 acc_test: 0.7740 time: 0.1453s
Epoch: 0411 loss_train: 0.0036 acc_train: 1.0000 loss_val: 0.8637 acc_val: 0.7000 loss_test: 0.7581 acc_test: 0.7740 time: 0.1016s
Epoch: 0421 loss_train: 0.0034 acc_train: 1.0000 loss_val: 0.8662 acc_val: 0.7000 loss_test: 0.7550 acc_test: 0.7690 time: 0.1607s
Epoch: 0431 loss_train: 0.0038 acc_train: 1.0000 loss_val: 0.8567 acc_val: 0.7100 loss_test: 0.7421 acc_test: 0.7720 time: 0.1545s
Epoch: 0441 loss_train: 0.0029 acc_train: 1.0000 loss_val: 0.8526 acc_val: 0.7033 loss_test: 0.7521 acc_test: 0.7840 time: 0.1279s
Epoch: 0451 loss_train: 0.0033 acc_train: 1.0000 loss_val: 0.8880 acc_val: 0.7067 loss_test: 0.7651 acc_test: 0.7670 time: 0.1217s
Epoch: 0461 loss_train: 0.0034 acc_train: 1.0000 loss_val: 0.8712 acc_val: 0.7033 loss_test: 0.7600 acc_test: 0.7780 time: 0.1688s
Epoch: 0471 loss_train: 0.0035 acc_train: 1.0000 loss_val: 0.8722 acc_val: 0.7033 loss_test: 0.7667 acc_test: 0.7720 time: 0.1191s
Epoch: 0481 loss_train: 0.0030 acc_train: 1.0000 loss_val: 0.8674 acc_val: 0.6967 loss_test: 0.7519 acc_test: 0.7720 time: 0.1045s
Epoch: 0491 loss_train: 0.0034 acc_train: 1.0000 loss_val: 0.8576 acc_val: 0.7033 loss_test: 0.7498 acc_test: 0.7790 time: 0.1683s
Epoch: 0501 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.8555 acc_val: 0.7000 loss_test: 0.7498 acc_test: 0.7840 time: 0.1211s
Epoch: 0511 loss_train: 0.0047 acc_train: 1.0000 loss_val: 0.8650 acc_val: 0.7033 loss_test: 0.7593 acc_test: 0.7710 time: 0.1185s
Epoch: 0521 loss_train: 0.0033 acc_train: 1.0000 loss_val: 0.8844 acc_val: 0.7100 loss_test: 0.7796 acc_test: 0.7760 time: 0.1714s
Epoch: 0531 loss_train: 0.0035 acc_train: 1.0000 loss_val: 0.8696 acc_val: 0.7100 loss_test: 0.7617 acc_test: 0.7810 time: 0.1375s
Epoch: 0541 loss_train: 0.0033 acc_train: 1.0000 loss_val: 0.8704 acc_val: 0.7133 loss_test: 0.7609 acc_test: 0.7710 time: 0.1246s
Epoch: 0551 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.8734 acc_val: 0.7200 loss_test: 0.7524 acc_test: 0.7780 time: 0.1165s
Epoch: 0561 loss_train: 0.0029 acc_train: 1.0000 loss_val: 0.8724 acc_val: 0.6967 loss_test: 0.7631 acc_test: 0.7760 time: 0.1222s
Epoch: 0571 loss_train: 0.0036 acc_train: 1.0000 loss_val: 0.8840 acc_val: 0.7033 loss_test: 0.7818 acc_test: 0.7740 time: 0.1202s
Epoch: 0581 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.8803 acc_val: 0.7133 loss_test: 0.7648 acc_test: 0.7830 time: 0.0905s
Epoch: 0591 loss_train: 0.0029 acc_train: 1.0000 loss_val: 0.8695 acc_val: 0.7133 loss_test: 0.7882 acc_test: 0.7820 time: 0.1235s
Epoch: 0601 loss_train: 0.0034 acc_train: 1.0000 loss_val: 0.8589 acc_val: 0.7100 loss_test: 0.7645 acc_test: 0.7830 time: 0.1752s
Epoch: 0611 loss_train: 0.0034 acc_train: 1.0000 loss_val: 0.8706 acc_val: 0.7100 loss_test: 0.7609 acc_test: 0.7720 time: 0.1050s
Epoch: 0621 loss_train: 0.0026 acc_train: 1.0000 loss_val: 0.8939 acc_val: 0.7033 loss_test: 0.7848 acc_test: 0.7710 time: 0.1725s
Epoch: 0631 loss_train: 0.0035 acc_train: 1.0000 loss_val: 0.8781 acc_val: 0.7133 loss_test: 0.7734 acc_test: 0.7780 time: 0.1215s
Epoch: 0641 loss_train: 0.0042 acc_train: 1.0000 loss_val: 0.8849 acc_val: 0.7133 loss_test: 0.7646 acc_test: 0.7720 time: 0.1554s
Epoch: 0651 loss_train: 0.0030 acc_train: 1.0000 loss_val: 0.8747 acc_val: 0.7200 loss_test: 0.7574 acc_test: 0.7740 time: 0.1191s
Epoch: 0661 loss_train: 0.0029 acc_train: 1.0000 loss_val: 0.8543 acc_val: 0.7033 loss_test: 0.7774 acc_test: 0.7820 time: 0.1724s
Epoch: 0671 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.8864 acc_val: 0.7200 loss_test: 0.7700 acc_test: 0.7710 time: 0.1193s
Epoch: 0681 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.9076 acc_val: 0.7033 loss_test: 0.7989 acc_test: 0.7650 time: 0.1708s
Epoch: 0691 loss_train: 0.0032 acc_train: 1.0000 loss_val: 0.8803 acc_val: 0.7133 loss_test: 0.7753 acc_test: 0.7770 time: 0.1102s
Epoch: 0701 loss_train: 0.0026 acc_train: 1.0000 loss_val: 0.8579 acc_val: 0.7267 loss_test: 0.7438 acc_test: 0.7790 time: 0.1653s
Epoch: 0711 loss_train: 0.0025 acc_train: 1.0000 loss_val: 0.8933 acc_val: 0.7200 loss_test: 0.7734 acc_test: 0.7710 time: 0.1604s
Epoch: 0721 loss_train: 0.0036 acc_train: 1.0000 loss_val: 0.8813 acc_val: 0.7033 loss_test: 0.7698 acc_test: 0.7730 time: 0.1270s
Epoch: 0731 loss_train: 0.0032 acc_train: 1.0000 loss_val: 0.8940 acc_val: 0.7200 loss_test: 0.7717 acc_test: 0.7690 time: 0.1250s
Epoch: 0741 loss_train: 0.0033 acc_train: 1.0000 loss_val: 0.8893 acc_val: 0.7100 loss_test: 0.7808 acc_test: 0.7760 time: 0.1273s
Epoch: 0751 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.9017 acc_val: 0.7100 loss_test: 0.7658 acc_test: 0.7710 time: 0.1183s
Epoch: 0761 loss_train: 0.0040 acc_train: 1.0000 loss_val: 0.9116 acc_val: 0.7067 loss_test: 0.7935 acc_test: 0.7640 time: 0.1203s
Epoch: 0771 loss_train: 0.0030 acc_train: 1.0000 loss_val: 0.8640 acc_val: 0.7000 loss_test: 0.7646 acc_test: 0.7840 time: 0.1202s
Epoch: 0781 loss_train: 0.0033 acc_train: 1.0000 loss_val: 0.9273 acc_val: 0.6933 loss_test: 0.8438 acc_test: 0.7570 time: 0.1045s
Epoch: 0791 loss_train: 0.0029 acc_train: 1.0000 loss_val: 0.9039 acc_val: 0.7067 loss_test: 0.7864 acc_test: 0.7790 time: 0.1214s
Epoch: 0801 loss_train: 0.0033 acc_train: 1.0000 loss_val: 0.8976 acc_val: 0.7067 loss_test: 0.7767 acc_test: 0.7760 time: 0.1189s
Epoch: 0811 loss_train: 0.0028 acc_train: 1.0000 loss_val: 0.8880 acc_val: 0.7267 loss_test: 0.7688 acc_test: 0.7790 time: 0.1272s
Epoch: 0821 loss_train: 0.0025 acc_train: 1.0000 loss_val: 0.8648 acc_val: 0.7100 loss_test: 0.7512 acc_test: 0.7880 time: 0.1695s
Epoch: 0831 loss_train: 0.0031 acc_train: 1.0000 loss_val: 0.9275 acc_val: 0.7033 loss_test: 0.8012 acc_test: 0.7610 time: 0.1658s
Epoch: 0841 loss_train: 0.0063 acc_train: 1.0000 loss_val: 0.9068 acc_val: 0.7067 loss_test: 0.7907 acc_test: 0.7700 time: 0.1224s
Epoch: 0851 loss_train: 0.0032 acc_train: 1.0000 loss_val: 1.0350 acc_val: 0.6967 loss_test: 0.9377 acc_test: 0.7550 time: 0.1825s
Epoch: 0861 loss_train: 0.0023 acc_train: 1.0000 loss_val: 0.9901 acc_val: 0.7033 loss_test: 0.8373 acc_test: 0.7700 time: 0.1656s
Epoch: 0871 loss_train: 0.0031 acc_train: 1.0000 loss_val: 0.9167 acc_val: 0.7100 loss_test: 0.8019 acc_test: 0.7690 time: 0.1679s
Epoch: 0881 loss_train: 0.0027 acc_train: 1.0000 loss_val: 0.8762 acc_val: 0.7267 loss_test: 0.7603 acc_test: 0.7770 time: 0.1250s
Epoch: 0891 loss_train: 0.0030 acc_train: 1.0000 loss_val: 0.9250 acc_val: 0.6967 loss_test: 0.8203 acc_test: 0.7590 time: 0.1184s
Epoch: 0901 loss_train: 0.0026 acc_train: 1.0000 loss_val: 0.8982 acc_val: 0.6933 loss_test: 0.7981 acc_test: 0.7680 time: 0.1238s
Epoch: 0911 loss_train: 0.0026 acc_train: 1.0000 loss_val: 0.8814 acc_val: 0.7033 loss_test: 0.7959 acc_test: 0.7800 time: 0.1689s
Epoch: 0921 loss_train: 0.0032 acc_train: 1.0000 loss_val: 0.8975 acc_val: 0.7067 loss_test: 0.7811 acc_test: 0.7690 time: 0.1774s
Epoch: 0931 loss_train: 0.0024 acc_train: 1.0000 loss_val: 0.8928 acc_val: 0.7133 loss_test: 0.7703 acc_test: 0.7790 time: 0.1185s
Epoch: 0941 loss_train: 0.0031 acc_train: 1.0000 loss_val: 0.9134 acc_val: 0.7100 loss_test: 0.7839 acc_test: 0.7670 time: 0.1680s
Epoch: 0951 loss_train: 0.0029 acc_train: 1.0000 loss_val: 0.9012 acc_val: 0.7067 loss_test: 0.7790 acc_test: 0.7700 time: 0.1748s
Epoch: 0961 loss_train: 0.0030 acc_train: 1.0000 loss_val: 0.9019 acc_val: 0.7267 loss_test: 0.7670 acc_test: 0.7720 time: 0.1661s
Epoch: 0971 loss_train: 0.0031 acc_train: 1.0000 loss_val: 0.9027 acc_val: 0.7033 loss_test: 0.7876 acc_test: 0.7610 time: 0.1197s
Epoch: 0981 loss_train: 0.0026 acc_train: 1.0000 loss_val: 0.8876 acc_val: 0.7033 loss_test: 0.7813 acc_test: 0.7720 time: 0.1203s
Epoch: 0991 loss_train: 0.0031 acc_train: 1.0000 loss_val: 0.8964 acc_val: 0.7133 loss_test: 0.7897 acc_test: 0.7700 time: 0.1223s
Optimization Finished!
Total time elapsed: 140.0382s, best testing performance  0.804000, minimun loss  0.665190
************ Finish ************
```