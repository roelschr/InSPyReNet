from utils.misc import *
from run import *
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    
    ray.init(
        runtime_env={
            "pip": "requirements.txt",
        },
    )
 
    scaling_config = ScalingConfig(num_workers=opt.Train.get("Workers", 2), use_gpu=True)

    run_config = RunConfig(storage_path="/mnt/cluster_storage/insyprenet/training", name="ray-train-inspyrenet")

    trainer = TorchTrainer(
        train,
        train_loop_config={"args": args, "opt": opt},
        scaling_config=scaling_config,
        run_config=run_config
    )
    result = trainer.fit()

    if args.local_rank <= 0:
        test(opt, args)
        evaluate(opt, args)
