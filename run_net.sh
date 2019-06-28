source activate tensorflow_p36
python GMF_keras.py --batch_size 256 --lr 0.01 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 256 --lr 0.005 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 256 --lr 0.001 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 512 --lr 0.01 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 512 --lr 0.005 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 512 --lr 0.001 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 1024 --lr 0.01 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 1024 --lr 0.005 --n_emb 8 --epochs 30
python GMF_keras.py --batch_size 1024 --lr 0.001 --n_emb 8 --epochs 30

# batch_size 1024 and lr 0.001
python GMF_keras.py --batch_size 1024 --lr 0.001 --n_emb 16 --epochs 30
python GMF_keras.py --batch_size 1024 --lr 0.001 --n_emb 32 --epochs 30
python GMF_keras.py --batch_size 1024 --lr 0.001 --n_emb 64 --epochs 30

source activate pytorch_p36
python GMF_pytorch.py --batch_size 256 --lr 0.01 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 256 --lr 0.005 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 256 --lr 0.001 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 512 --lr 0.01 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 512 --lr 0.005 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 512 --lr 0.001 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 1024 --lr 0.01 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 1024 --lr 0.005 --n_emb 8 --epochs 30
python GMF_pytorch.py --batch_size 1024 --lr 0.001 --n_emb 8 --epochs 30

# batch_size 256 and lr 0.001
python GMF_pytorch.py --batch_size 256 --lr 0.001 --n_emb 16 --epochs 30
python GMF_pytorch.py --batch_size 256 --lr 0.001 --n_emb 32 --epochs 30
python GMF_pytorch.py --batch_size 256 --lr 0.001 --n_emb 64 --epochs 30

source activate amazonei_mxnet_p36
python GMF_gluon.py --batch_size 256 --lr 0.01 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 256 --lr 0.005 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 256 --lr 0.001 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 512 --lr 0.01 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 512 --lr 0.005 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 512 --lr 0.001 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 1024 --lr 0.01 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 1024 --lr 0.005 --n_emb 8 --epochs 30
python GMF_gluon.py --batch_size 1024 --lr 0.001 --n_emb 8 --epochs 30

# batch_size 1024 and lr 0.001
python GMF_gluon.py --batch_size 1024 --lr 0.001 --n_emb 16 --epochs 30
python GMF_gluon.py --batch_size 1024 --lr 0.001 --n_emb 32 --epochs 30
python GMF_gluon.py --batch_size 1024 --lr 0.001 --n_emb 64 --epochs 30

source activate tensorflow_p36
python MLP_keras.py --batch_size 256 --lr 0.001 --layers "[32, 16, 8]" --epochs 30
python MLP_keras.py --batch_size 512 --lr 0.001 --layers "[32, 16, 8]" --epochs 30
python MLP_keras.py --batch_size 1024 --lr 0.001 --layers "[32, 16, 8]" --epochs 30

python MLP_keras.py --batch_size 256 --lr 0.001 --layers "[64, 32, 16]" --epochs 30
python MLP_keras.py --batch_size 256 --lr 0.001 --layers "[128, 64, 32]" --epochs 30
python MLP_keras.py --batch_size 256 --lr 0.001 --layers "[256, 128, 64]" --epochs 30

source activate pytorch_p36
python MLP_pytorch.py --batch_size 256 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python MLP_pytorch.py --batch_size 256 --lr 0.001 --layers "[32, 16, 8]" --epochs 30
python MLP_pytorch.py --batch_size 512 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python MLP_pytorch.py --batch_size 512 --lr 0.001 --layers "[32, 16, 8]" --epochs 30
python MLP_pytorch.py --batch_size 1024 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python MLP_pytorch.py --batch_size 1024 --lr 0.001 --layers "[32, 16, 8]" --epochs 30

python MLP_pytorch.py --batch_size 256 --lr 0.01 --layers "[64, 32, 16]" --epochs 30
python MLP_pytorch.py --batch_size 256 --lr 0.01 --layers "[128, 64, 32]" --epochs 30
python MLP_pytorch.py --batch_size 256 --lr 0.01 --layers "[256, 128, 64]" --epochs 30

source activate amazonei_mxnet_p36
python MLP_gluon.py --batch_size 256 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python MLP_gluon.py --batch_size 256 --lr 0.001 --layers "[32, 16, 8]" --epochs 30
python MLP_gluon.py --batch_size 512 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python MLP_gluon.py --batch_size 512 --lr 0.001 --layers "[32, 16, 8]" --epochs 30
python MLP_gluon.py --batch_size 1024 --lr 0.01 --layers "[32, 16, 8]" --epochs 30
python MLP_gluon.py --batch_size 1024 --lr 0.001 --layers "[32, 16, 8]" --epochs 30

python MLP_gluon.py --batch_size 512 --lr 0.01 --layers "[64, 32, 16]" --epochs 30
python MLP_gluon.py --batch_size 512 --lr 0.01 --layers "[128, 64, 32]" --epochs 30
python MLP_gluon.py --batch_size 512 --lr 0.01 --layers "[256, 128, 64]" --epochs 30

source activate tensorflow_p36
python NeuMF_keras.py --batch_size 1024 --lr 0.0001 --n_emb 16 --layers "[256, 128, 64]" \
--mf_pretrain "keras_GMF_bs_1024_reg_00_lr_0001_n_emb_16.h5" \
--mlp_pretrain "keras_MLP_bs_256_reg_00_lr_0001_n_emb_128_ll_64_dp_wodp.h5" \
--learner "SGD" --epochs 10
python NeuMF_keras.py --batch_size 1024 --lr 0.0001 --n_emb 16 --layers "[256, 128, 64]" \
--mf_pretrain "keras_GMF_bs_1024_reg_00_lr_0001_n_emb_16.h5" \
--mlp_pretrain "keras_MLP_bs_256_reg_00_lr_0001_n_emb_128_ll_64_dp_wodp.h5" \
--freeze 1 --learner "SGD" --epochs 10

source activate pytorch_p36
python NeuMF_pytorch.py --batch_size 1024 --lr 0.001 --n_emb 32 --layers "[128, 64, 32]" \
--mf_pretrain "pytorch_GMF_bs_256_lr_0001_n_emb_32.pt" \
--mlp_pretrain "pytorch_MLP_bs_256_reg_00_lr_001_n_emb_64_ll_32_dp_wodp.pt" \
--learner "SGD" --epochs 10
python NeuMF_pytorch.py --batch_size 1024 --lr 0.001 --n_emb 32 --layers "[128, 64, 32]" \
--mf_pretrain "pytorch_GMF_bs_256_lr_0001_n_emb_32.pt" \
--mlp_pretrain "pytorch_MLP_bs_256_reg_00_lr_001_n_emb_64_ll_32_dp_wodp.pt" \
--freeze 1 --learner "SGD" --epochs 10

source activate amazonei_mxnet_p36
python NeuMF_gluon.py --batch_size 1024 --lr 0.001 --n_emb 32 --layers "[256, 128, 64]" \
--mf_pretrain "gluon_GMF_bs_1024_lr_0001_n_emb_32.params" \
--mlp_pretrain "gluon_MLP_bs_512_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.params" \
--learner "SGD" --epochs 10
python NeuMF_gluon.py --batch_size 1024 --lr 0.001 --n_emb 32 --layers "[256, 128, 64]" \
--mf_pretrain "gluon_GMF_bs_1024_lr_0001_n_emb_32.params" \
--mlp_pretrain "gluon_MLP_bs_512_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.params" \
--freeze 1 --learner "SGD" --epochs 10