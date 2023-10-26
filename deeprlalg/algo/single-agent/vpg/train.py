from vpg import VPG
from deeprlalg.user_config import DEFAULT_DATA_DIR
import os
import time

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--epochs', '-n', type=int, default=200)

    #Advantage Estimation Parameters
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--gae_lambda', type=float, default=None)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--test_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=1000) ##steps used per gradient step
    parser.add_argument('--render', '-r', type=bool, default=False)


    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--seed', type=int,  nargs='+', default=[1])

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--action_noise_std', type=float, default=0)

    args = parser.parse_args()
    seeds = args.seed
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = params['data_dir'] or DEFAULT_DATA_DIR
    logdir = args.exp_name + '_' + args.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    for i in range(len(seeds)):
        args.seed = seeds[i]
        params['seed'] = seeds[i]
        params['train_batch_size'] = params['batch_size'] # ??
        seedlogdir = os.path.join(logdir, str(params['seed']))
        params['logdir'] = seedlogdir
        seedparams = params.copy()

        ###################
        ### RUN TRAINING
        ###################

        algo = VPG(seedparams)
        algo.run()


if __name__ == "__main__":
    main()
