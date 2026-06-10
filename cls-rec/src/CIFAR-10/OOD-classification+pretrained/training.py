import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Discriminator, Encoder, Generator
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP

def evaluate(encoder, discriminator, id_loader, ood_loader):
    id_scores  = []
    ood_scores = []

    with torch.inference_mode():
        for images, _ in id_loader:
            images = images.cuda()

            d_logit = discriminator(encoder(images, mean=True)[0])
            ood_score = torch.sigmoid(d_logit)
            id_scores.append(ood_score.detach().flatten().cpu())

        id_scores = torch.cat(id_scores, dim=0).numpy()

        print('')

        for images, _ in ood_loader:
            images = images.cuda()

            d_logit = discriminator(encoder(images, mean=True)[0])
            ood_score = torch.sigmoid(d_logit)
            ood_scores.append(ood_score.detach().flatten().cpu())

            print(len(ood_scores) * len(ood_scores[0]), '/', 50000, end='\r')

        ood_scores = torch.cat(ood_scores, dim=0).numpy()

    labels = np.concatenate([np.zeros_like(id_scores, dtype=np.int64), np.ones_like(ood_scores, dtype=np.int64)])
    labels = np.asarray(labels).reshape(-1)
    scores = np.concatenate([id_scores, ood_scores])    
    scores = np.asarray(scores).reshape(-1)

    labels_in = 1 - labels
    scores_in = -scores

    AUROC   = float(roc_auc_score(labels, scores) * 100.0)
    AUPR_IN = float(average_precision_score(labels_in, scores_in) * 100.0)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    FPR95 = float(1.0 if len(idx) == 0 else float(fpr[idx[0]]) * 100)

    return AUROC, AUPR_IN, FPR95

if __name__ == '__main__':
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])

    batch_size  = 16
    iterations  = 1000000
    D_model     = Discriminator().cuda(local_rank)
    E_model     = Encoder(pretrained=True).cuda(local_rank)
    G_model     = Generator().cuda(local_rank)
    D           = DDP(D_model, device_ids=[local_rank], output_device=local_rank)
    E           = DDP(E_model, device_ids=[local_rank], output_device=local_rank)
    G           = DDP(G_model, device_ids=[local_rank], output_device=local_rank)
    D_optimizer = optim.Adam(list(D.parameters()), lr=0.0001, betas=(0.5, 0.9), weight_decay=1e-6)
    G_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=0.0001, betas=(0.5, 0.9), weight_decay=1e-6)
    loss        = [None] * 10
    i           = 0

    '''
    checkpoint = torch.load('20000/checkpoint.pt', map_location='cpu')
    D_model.load_state_dict(checkpoint['D_model'])
    E_model.load_state_dict(checkpoint['E_model'])
    G_model.load_state_dict(checkpoint['G_model'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer'])
    checkpoint = None
    '''

    transform   = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset    = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    sampler     = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True, sampler=sampler, num_workers=2, pin_memory=True)
    transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset     = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, pin_memory=True)
    transform   = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(32), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    oodset      = torchvision.datasets.ImageFolder(root='../../ImageNet/data/validation', transform=transform)
    oodloader   = torch.utils.data.DataLoader(oodset, batch_size=256, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    try:
        while i < iterations:
            sampler.set_epoch(i)

            for j, (image, label) in enumerate(trainloader):
                if i >= iterations:
                    break

                image = image.cuda(local_rank, non_blocking=True)
                input = image.detach().requires_grad_(True)
                label = label.cuda(local_rank, non_blocking=True)
                noise = torch.randn(size=(batch_size, 512), dtype=torch.float32).cuda()
                noise = noise.detach().requires_grad_(True)

                latent, infer = E(image)

                real_density = D(noise)
                fake_density = D(latent.detach())

                gradient = autograd.grad(outputs=real_density, inputs=[noise], grad_outputs=torch.ones(real_density.size()).cuda(local_rank), create_graph=True, retain_graph=True)
                gradient = torch.cat([
                        gradient[0].contiguous().view(input.size(0), -1),
                ], dim=1)
                
                D_optimizer.zero_grad()
                D_loss = [F.softplus(-real_density).mean(), F.softplus(fake_density).mean(), 0.5 * (((gradient.norm(2, dim=1) - 0) ** 2)).mean()]
                (D_loss[0] + D_loss[1] + 0.1 * D_loss[2]).backward()
                D_optimizer.step()
                
                loss[0] = D_loss[0].item()
                loss[1] = D_loss[1].item()
                loss[2] = D_loss[2].item()

                if i % 5 == 0:
                    G_optimizer.zero_grad()
                    G_loss = [F.softplus(-D(latent)).mean(), nn.CrossEntropyLoss()(G(latent), label), nn.CrossEntropyLoss()(infer, label)]
                    (G_loss[0] + 10 * (G_loss[1] + G_loss[2])).backward()
                    G_optimizer.step()

                    loss[3] = G_loss[0].item()
                    loss[4] = G_loss[1].item()
                    loss[5] = G_loss[2].item()

                if rank == 0:
                    print(i + 1, '/', iterations, '\tD:', loss[0], loss[1], loss[2], '\tG:', loss[3], loss[4], loss[5], end='\r')

                if rank == 0 and ((i + 1) % 10000 == 0 or i == 0):
                    D_model.eval()                    
                    E_model.eval()
                    
                    AUROC, AUPR_IN, FPR95 = evaluate(encoder=E_model, discriminator=D_model, id_loader=testloader, ood_loader=oodloader)
                    print('AUROC:', AUROC, '\tAUPR_IN:', AUPR_IN, '\tFPR@95:', FPR95)
                    
                    if not os.path.exists('{}'.format(i + 1)):
                        os.makedirs('{}'.format(i + 1))
                        
                    torch.save({
                            'D_model': D_model.state_dict(),
                            'E_model': E_model.state_dict(),
                            'G_model': G_model.state_dict(),
                            'D_optimizer': D_optimizer.state_dict(),
                            'G_optimizer': G_optimizer.state_dict(),
                            }, '{}/checkpoint.pt'.format(i + 1))

                    D_model.train()                    
                    E_model.train()

                dist.barrier()
                i += 1
    finally:
        dist.destroy_process_group()
