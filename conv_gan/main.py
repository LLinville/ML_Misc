import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
def main(config):
    cudnn.benchmark = True
    
    data_loader = get_loader(image_path=config.image_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers)
    
    solver = Solver(config, data_loader)
    
    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'sample':
        solver.sample()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    run = 5
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='I:/ml_save/dcgan/saves/run' + str(run) + '/model')
    parser.add_argument('--sample_path', type=str, default='I:/ml_save/dcgan/saves/run' + str(run))
    # parser.add_argument('--model_path', type=str, default='./saves/run' + str(run) + '/model')
    # parser.add_argument('--sample_path', type=str, default='.//saves/run' + str(run))
    # parser.add_argument('--image_path', type=str, default='../../../CelebA/128_crop')
    parser.add_argument('--image_path', type=str, default='C:/Users/Eracoy/PycharmProjects/CelebA/128_crop')
    # parser.add_argument('--image_path', type=str, default='I:/cartoonset100k')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=100)

    config = parser.parse_args()
    print(config)
    main(config)