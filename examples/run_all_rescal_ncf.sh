Datasets=("ml_100k" "ml_1m" "foursquare")
lambda_As=(1 5 10 20)
lambda_Rs=(1 5 10 20)
for Dataset in ${Datasets[*]}; do
    for lambda_A in ${lambda_As[*]}; do
        for lambda_R in ${lambda_Rs[*]}; do
            /home/zm324/anaconda3/envs/beta_rec/bin/python train_rescal_ncf.py --dataset $Dataset --lambda_A $lambda_A --lambda_R $lambda_R
        done
    done
done
for Dataset in ${Datasets[*]}; do
    for lambda_A in ${lambda_As[*]}; do
        for lambda_R in ${lambda_Rs[*]}; do
            /home/zm324/anaconda3/envs/beta_rec/bin/python train_rescal_ncf.py --dataset $Dataset --lambda_A $lambda_A --lambda_R $lambda_R
        done
    done
done
for Dataset in ${Datasets[*]}; do
    for lambda_A in ${lambda_As[*]}; do
        for lambda_R in ${lambda_Rs[*]}; do
            /home/zm324/anaconda3/envs/beta_rec/bin/python train_rescal_ncf.py --dataset $Dataset --lambda_A $lambda_A --lambda_R $lambda_R
        done
    done
done
for Dataset in ${Datasets[*]}; do
    for lambda_A in ${lambda_As[*]}; do
        for lambda_R in ${lambda_Rs[*]}; do
            /home/zm324/anaconda3/envs/beta_rec/bin/python train_rescal_ncf.py --dataset $Dataset --lambda_A $lambda_A --lambda_R $lambda_R
        done
    done
done
for Dataset in ${Datasets[*]}; do
    for lambda_A in ${lambda_As[*]}; do
        for lambda_R in ${lambda_Rs[*]}; do
            /home/zm324/anaconda3/envs/beta_rec/bin/python train_rescal_ncf.py --dataset $Dataset --lambda_A $lambda_A --lambda_R $lambda_R
        done
    done
done