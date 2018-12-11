import torch


def str2bool(s):
    return s.lower() in ('true', '1')


def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False
