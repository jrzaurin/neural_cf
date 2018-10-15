import pandas as pd

df = pd.read_pickle("results_df.p")

select_keras = df[df.modelname.str.contains("keras")].sort_values("best_hr", ascending=False)

#               keras_GMF_bs_256_reg_00_lr_0001_n_emb_32.h5  0.700166
# keras_MLP_bs_256_reg_00_lr_0001_n_emb_64_ll_32_dp_wodp.h5  0.686755

select_pytorch = df[df.modelname.str.contains("pytorch")].sort_values("best_hr", ascending=False)

# pytorch_MLP_bs_256_reg_00_lr_001_n_emb_128_ll_64_dp_wodp.pt  0.826656
                      # pytorch_GMF_bs_256_lr_001_n_emb_64.pt  0.801987

select_gluon = df[df.modelname.str.contains("gluon")].sort_values("best_hr", ascending=False)

# gluon_MLP_bs_256_reg_00_lr_001_n_emb_32_ll_16_dp_wodp.params  0.936093
                    # gluon_GMF_bs_1024_lr_001_n_emb_16.params  0.681291