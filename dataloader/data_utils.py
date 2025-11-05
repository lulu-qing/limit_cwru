import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# def set_up_datasets(args):
#     if args.dataset == 'cifar100':
#         import dataloader.cifar100.cifar as Dataset
#         args.base_class = 60
#         args.num_classes=100
#         args.way = 5
#         args.shot = 5
#         args.sessions = 9

#     args.Dataset=Dataset
#     return args

# def get_dataloader(args,session):
#     if session == 0:
#         trainset, trainloader, testloader = get_base_dataloader(args)
#     else:
#         trainset, trainloader, testloader = get_new_dataloader(args)
#     return trainset, trainloader, testloader

# def get_base_dataloader(args):
#     txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
#     class_index = np.arange(args.base_class)
#     if args.dataset == 'cifar100':

#         trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
#                                          index=class_index, base_sess=True)
#         testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
#                                         index=class_index, base_sess=True)
        
#         print(len(trainset), len(testset))

#     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
#                                               num_workers=0, pin_memory=True)
#     testloader = torch.utils.data.DataLoader(
#         dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

#     return trainset, trainloader, testloader



# def get_base_dataloader_meta(args):
#     txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
#     class_index = np.arange(args.base_class)
#     if args.dataset == 'cifar100':
#         trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
#                                          index=class_index, base_sess=True)
#         testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
#                                         index=class_index, base_sess=True)



#     # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
#     sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
#                                 args.episode_shot + args.episode_query)

#     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
#                                               pin_memory=True)

#     testloader = torch.utils.data.DataLoader(
#         dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

#     return trainset, trainloader, testloader

# def get_new_dataloader(args,session):
#     txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
#     if args.dataset == 'cifar100':
#         class_index = open(txt_path).read().splitlines()
#         # print(class_index)
#         trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
#                                          index=class_index, base_sess=False)

#     if args.batch_size_new == 0:
#         batch_size_new = trainset.__len__()
#         # print(batch_size_new)
#         trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
#                                                   num_workers=args.num_workers, pin_memory=True)
#     else:
#         # print(batch_size_new)
#         trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
#                                                   num_workers=args.num_workers, pin_memory=True)

#     # test on all encountered classes
#     class_new = get_session_classes(args, session)

#     if args.dataset == 'cifar100':
#         testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
#                                         index=class_new, base_sess=False)
    

#     testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
#                                              num_workers=args.num_workers, pin_memory=True)

#     return trainset, trainloader, testloader

# def get_session_classes(args,session):
#     class_list=np.arange(args.base_class + session * args.way)
#     return class_list

def set_up_datasets(args):
    if args.dataset == 'cwru':
        import dataloader.cwru as Dataset
        args.base_class = 4
        args.num_classes=10
        args.way = 1
        args.shot = 5
        args.sessions = 7

    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cwru':

        trainset = args.Dataset.CWRU_dataset(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.CWRU_dataset(root=args.dataroot, train=False, index=class_index, base_sess=True)
        
        print(len(trainset), len(testset))

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cwru':
        trainset = args.Dataset.CWRU_dataset(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CWRU_dataset(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)



    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader



def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cwru':
        class_index = open(txt_path).read().splitlines()
        # print(class_index)
        trainset = args.Dataset.CWRU_dataset(root=args.dataroot, train=True, index=class_index, base_sess=False)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        # print(batch_size_new)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        # print(batch_size_new)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cwru':
        testset = args.Dataset.CWRU_dataset(root=args.dataroot, train=False, index=class_new, base_sess=False)

    

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list
