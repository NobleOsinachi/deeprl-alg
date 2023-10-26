from maddpg import MADDPG
from deeprlalg.user_config import DEFAULT_DATA_DIR
import os
import time

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--algo', type=str, default='maddpg')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--epochs', '-n', type=int, default=200)
    parser.add_argument('--steps_per_epoch', '-spe', type=int, default=4000)


    parser.add_argument('--buffer_size', '-bs', type=int, default=1e6)

    parser.add_argument('--polyak', '-pk', type=int, default=0.995)
    parser.add_argument('--gamma', '-g', type=int, default=0.99)



    #actor parameters
    parser.add_argument('--pi_lr', '-plr', type=int, default=1e-3)
    parser.add_argument('--pi_output_activation', '-p_oact', type=str, default='identity')
    parser.add_argument('--pi_activation', '-p_act', type=str, default='tanh')
    parser.add_argument('--pi_hidden_sizes', '-p_hs', nargs='+', default=[256, 256])

    #q parameters
    parser.add_argument('--q_lr', '-qlr', type=int, default=1e-3)
    parser.add_argument('--q_output_activation', '-q_oact', type=str, default='identity')
    parser.add_argument('--q_activation', '-q_act', type=str, default='tanh')
    parser.add_argument('--q_hidden_sizes', '-q_hs', nargs='+', default=[256, 256])

    #Training parameters
    parser.add_argument('--batch_size', '-b', type=int, default=100) #steps collected per train iteration
    parser.add_argument('--start_steps', '-ss', type=int, default=10000)
    parser.add_argument('--update_after', '-ua', type=int, default=1000)
    parser.add_argument('--update_freq', '-uf', type=int, default=50)
    parser.add_argument('--action_noise', '-an', type=float, default=0.1)
    parser.add_argument('--num_test_episodes', '-num_test', type=int, default=10)
    parser.add_argument('--max_path_length', '-mpl', type=int, default=100000)

    parser.add_argument('--save_freq', '-sf', type=int, default=1)
    parser.add_argument('--render', '-r', type=bool, default=False)



    parser.add_argument('--seed', type=int,  nargs='+', default=[1])

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
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
    print("data+path: ", data_path)
    logdir = args.exp_name + '_' + args.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    for i in range(len(seeds)):
        args.seed = seeds[i]
        params['seed'] = seeds[i]
        seedlogdir = os.path.join(logdir, str(params['seed']))
        params['logdir'] = seedlogdir
        seedparams = params.copy()

        ###################
        ### RUN TRAINING
        ###################

        algo = MADDPG(seedparams)
        algo.run()


if __name__ == "__main__":
    main()
