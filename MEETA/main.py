from utils import *
from Trainer import *
from config import *
import time


parser = argparse.ArgumentParser(description="Script for processing configuration file")
parser.add_argument('--device', type=int, default=0, help='GPU device id')
parser.add_argument('--case', type=int, default=0, help='Select any case (1-6)')
parser.add_argument('--play', type=str, default='train', help='train or integration')
parser.add_argument('--dataset', type=str, default='DAVIS', help='DAVIS or KIBA')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args_dict = args.__dict__

args_dict.update(configuration)



if 0 <args.case < 7:
    args_dict.update(cases[args.case-1])
elif args.play != 'train':
    pass
else:
    raise Exception("Please select case number in 1-6")




config = dict2namespace(args_dict)

# add device
device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

print(config)

if __name__ == "__main__":
    
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    runner = WorkStation(config= config,
                     device= device)
    
    
    
    if config.play == 'train':
        runner.train()
    
    elif config.play == 'integration':
        runner.integration()
    
    else:
        raise Exception("no command")