import torch
from utils import get_loaders, eval_seg, eval_det, eval_cls
from models.DLHW2Net_v2 import DLHW2Net
import json

def evaluate(model_path, data_root, task, batch_size=16, n_workers=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = DLHW2Net()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()

    eval_funcs = {'seg': eval_seg, 'det': eval_det, 'cls': eval_cls}
    task_eval_dict = {'seg': 'mIoU', 'det': 'mAP', 'cls': 'Top-1 acc'}

    if task != 'all':
        if task not in eval_funcs:
            raise ValueError(f"Unknown task: {task}")
        loader = get_loaders(task, batch_size, n_workers, data_root)['val']
        res = eval_funcs[task](net, loader, device)
        return {task_eval_dict[task]: res}

    # Evaluate all tasks on validation splits
    results = {}
    for t in ['seg', 'det', 'cls']:
        loader = get_loaders(t, batch_size, n_workers, data_root, device)['val']
        metrics = eval_funcs[t](net, loader, device)
        results[task_eval_dict[t]] = metrics
    print(results)
            
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type = str, required = True)
    p.add_argument('--data_root', type=str, required = True)
    p.add_argument('--task', type=str, required=True, choices=['seg','det','cls','all'])
    p.add_argument('--batch_size', type = int, default = 16)
    p.add_argument('--n_workers', type = int, default = 5)
    args = p.parse_args()
    evaluate(args.weights, args.data_root, args.task, args.batch_size, args.n_workers)
