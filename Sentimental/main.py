from train import sweep_train
import wandb
import argparse

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description="Sentimental anlysis")
    parser.add_argument("--train", type=bool, default = False)
    parser.add_argument("--inference", type=bool, default = False)
    args = parser.parse_args()
    
    wandb_config = {
        'method': 'grid',
        'parameters':
        {
            'max_length': {'values' : [500, 250]},
        }
    }

    if args.train :
        sweep_id = wandb.sweep(sweep=wandb_config,
                        project='sweep_merged_all'
                        )
        wandb.agent(sweep_id, function=sweep_train)
        
    elif args.inference :
        # TODO: inference 코드 구현
        pass