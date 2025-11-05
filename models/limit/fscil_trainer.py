from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader

from .helper import *
from utils import *
from dataloader.data_utils import *
from dataloader.sampler import BasePreserverCategoriesSampler,NewCategoriesSampler
from .Network import MYNET


#copy from acastle.
class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        
        pass


#定义了一个 set_up_model 函数，目的是初始化并设置模型，加载预训练的模型权重（如果有的话），并进行必要的配置
    def set_up_model(self):
        #MYNET：这是一个模型类，可能是定义的神经网络模型的名称。它可能在其他地方已定义，包含了模型的具体架构（如卷积层、全连接层等）
        #mode=self.args.base_mode：这个参数设置模型的运行模式（如训练模式或推理模式）
        self.model = MYNET(self.args, mode=self.args.base_mode)
   
        #将模型包装为支持多GPU训练的 DataParallel 模型
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
    
    #如果 model_dir 参数不为 None，说明用户希望从指定的路径加载预训练的模型权重
        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)
            #该函数用于加载保存的模型的权重和各个参数
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            pass


    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, train_fsl_loader,train_gfsl_loader, testloader = self.get_base_dataloader_meta()
            return trainset, train_fsl_loader, train_gfsl_loader, testloader
        else:
            trainset, trainloader, testloader,train_fsl_loader = self.get_new_dataloader(session)
            return trainset, trainloader, testloader, train_fsl_loader


#创建并返回训练和测试数据集的 DataLoader
    def get_base_dataloader_meta(self):
        # sample 60 way 1 shot, 15query.
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
 
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cwru':
            # class_index = np.arange(self.args.base_class)
            #CWRU_dataset是cwru.py文件中的函数
            trainset = self.args.Dataset.CWRU_dataset(root=self.args.dataroot, train=True, index=class_index, base_sess=True)
            testset = self.args.Dataset.CWRU_dataset(root=self.args.dataroot, train=False, index=class_index, base_sess=True)
    
    #用了 DataLoader 的默认行为，即随机采样。每个批次从数据集中随机选取一部分数据，按照 batch_size 来组织成一个批次，train_gfsl_loader 主要加载查询集数据（query data）
        train_gfsl_loader = DataLoader(dataset=trainset, 
                                   batch_size=self.args.batch_size_base, 
                                   shuffle=True, 
                                   num_workers=0,
                                   pin_memory=True) 


        #创建了一个 CategoriesSampler，用于为训练过程中的每个批次按类别采样样本
        train_sampler = CategoriesSampler(trainset.targets,  len(train_gfsl_loader), self.args.sample_class, self.args.sample_shot)
        #创建了一个新的 DataLoader，使用之前定义的 train_sampler 来采样训练数据,也就是按照每个类别多少个样本来采样。train_fsl_loader 主要加载支持集数据（support data）。

        train_fsl_loader = DataLoader(dataset=trainset, 
                                    batch_sampler=train_sampler, 
                                    num_workers=0,
                                    pin_memory=True)  

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return trainset, train_fsl_loader,train_gfsl_loader, testloader


    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cwru':
    
    #打开上述生成的文本文件 txt_path，读取文件内容，并通过 splitlines() 方法按行分割，得到一个列表 class_index，这个列表包含了所有要加载的类的索引        
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CWRU_dataset(root=self.args.dataroot, train=True, index=class_index, base_sess=False)
     #如果 batch_size_new 为 0，则将批量大小设置为整个训练集的大小（即每次加载所有样本）。这是为了保证每个会话中可以加载所有的训练数据   
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            #trainloader 是为一般训练设计的普遍数据加载器，用于标准的 minibatch 训练
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=0, pin_memory=True)

    #如果 batch_size_new 不为 0，则使用提供的批量大小创建数据加载器 trainloader    
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=0, pin_memory=True)

    #NewCategoriesSampler：这个采样器用于为 Few-Shot 学习创建一个按类别随机抽样的训练加载器。它根据 trainset.targets（即训练集中的目标标签）创建一个新的样本批次    #1：表示迭代一次；1：表示每次迭代时的类别数是1；5：表示每个类别的样本数是5 
        test_sampler = NewCategoriesSampler(trainset.targets, 1, 1, 5)
       
       # batch_sampler=test_sampler 这个参数指定使用 test_sampler 作为批处理采样器,train_fsl_loader 是为少量样本学习量身定制的加载器，专注于从每个类别中选择样本，以实现快速学习和准确性
        train_fsl_loader = DataLoader(dataset=trainset,
                                    batch_sampler=test_sampler,
                                    num_workers=0,
                                    pin_memory=True)  
    
    #获取当前会话中测试时要使用的类别 class_new
        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cwru':
            #使用 CWRU_dataset 类来创建测试数据集 testset
            testset = self.args.Dataset.CWRU_dataset(root=self.args.dataroot, train=False, index=class_new, base_sess=False)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)

                  
        return trainset, trainloader, testloader, train_fsl_loader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list



#设置优化器（optimizer）和学习率调度器（scheduler），并返回它们
    def get_optimizer_base(self):
    #获取所有不包含 encoder 和 cls 关键字的模型参数
    #self.model.named_parameters()：返回模型中所有参数的名字和值
    #top_para 主要包含除了编码器（encoder）和分类头（cls）以外的参数
        top_para = [v for k,v in self.model.named_parameters() if ('encoder' not in k and 'cls' not in k)] 

        #设置一个基于 SGD（随机梯度下降）的优化器
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': top_para, 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

    #根据 schedule 参数选择学习率调度器
        if self.args.schedule == 'Step':
            #torch.optim.lr_scheduler.StepLR：定义了一个学习率调度器，每经过固定的步数（step_size），学习率就会按给定的衰减因子（gamma）降低
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        
        
        elif self.args.schedule == 'Milestone':
          #milestones=self.args.milestones：指定了在训练的哪个 epoch 对学习率进行调整。milestones 是一个列表，包含了所有需要进行学习率调整的 epoch  
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

               
        return optimizer, scheduler

    def train(self):
        args = self.args
        
        t_start_time = time.time()

        # init train statistics
        self.result_list = [args]

        for session in range(args.start_session, args.sessions):
            if session==0:
                train_set, train_fsl_loader, train_gfsl_loader, testloader = self.get_dataloader(session)
            else:
                train_set, trainloader, testloader, train_fsl_loader = self.get_dataloader(session)
            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.base_train(self.model, train_fsl_loader,train_gfsl_loader, optimizer, scheduler, epoch, args)
                    # print(f"Args before replace_base_fc: {args}")  # 打印 args
                    
                    # 将模型的全连接层的前base_class个权重替换成由train_set求得的原型
                    self.model = replace_base_fc(train_set, self.model, self.args)

                    self.model.module.mode = 'avg_cos'
            
            #这里检查是否设置了跳过验证过程（set_no_val）。如果是 True，那么在训练过程中不进行验证，只进行测试并保存模型
                    if args.set_no_val: # set no validation
                        #根据当前会话（session）的编号，构建保存模型的文件路径
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        #将当前模型的状态字典（state_dict）保存到指定路径 save_model_dir。状态字典包含模型的所有参数（权重和偏置）
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        #将优化器的状态字典保存到指定路径，这样在恢复训练时可以加载优化器的状态
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        #拷贝当前模型的状态字典，保存在 best_model_dict 中，以备后续使用
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        #调用 test() 函数来测试模型
                        tsl, tsa, true, pre = self.test(self.model, testloader, testloader, args, session)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        #获取当前学习率（lrc），通常通过学习率调度器（scheduler）来获取
                        lrc = scheduler.get_last_lr()[0]
                        print('\n epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                        self.result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                    else:
                        # take the last session's testloader for validation
                        vl, va, true, pre = self.validation()
                        # 如果当前验证准确率（va）大于或等于当前会话的最大准确率，则认为模型有所改进，更新最优模型
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                          self.trlog['max_acc'][session]))
                        self.trlog['val_loss'].append(vl)
                        self.trlog['val_acc'].append(va)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                        self.result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)

                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

         
                # 加载最好的模型参数（best_model_dict）到模型中
                self.model.load_state_dict(self.best_model_dict)
                #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                #再次构建最优模型保存路径
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                #保存当前模型的状态字典到最优模型路径
                torch.save(dict(params=self.model.state_dict()), best_model_dir)
         
                self.model.module.mode = 'avg_cos'
                #调用 test() 函数进行最终测试，返回测试损失（tsl）和测试准确率（tsa）
                tsl, tsa, true, pre = self.test(self.model, testloader, None, args, session)
                #更新当前会话的最大测试准确率
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                #打印当前会话的测试准确率
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
  
   
                #将当前会话的测试信息添加到 result_list 中
                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))



    #如果session不等于0则进入增量学习阶段
            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                
                #加载模型的最优参数字典（best_model_dict）到当前模型（self.model）
                self.model.load_state_dict(self.best_model_dict)
                #设置模型的模式（mode
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                # trainloader.dataset.transform = testloader.dataset.transform
                # train_fsl_loader.dataset.transform = testloader.dataset.transform
                

                #更新模型的全连接层（update_fc）。在增量学习中，可能会遇到新的类别或数据集，当前模型的最后一层（全连接层，fc）需要更新以适应新的类别或任务
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                tsl, tsa, true, pre = self.test(self.model, testloader,train_fsl_loader, args, session,validation=False)

                # save better model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        self.result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        self.result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

        return true, pre

    def validation(self):
        with torch.no_grad():
            model = self.model
            session=1
            #for session in range(1, self.args.sessions):
            trainset, trainloader, testloader,train_fsl_loader = self.get_dataloader(session)
            # trainloader.dataset.transform = testloader.dataset.transform
            # train_fsl_loader.dataset.transform = testloader.dataset.transform
            model.module.mode = 'avg_cos'
            model.eval()
            
            model.module.update_fc(trainloader, np.unique(trainset.targets), session)
            vl, va, true, pre = self.test(model, testloader,train_fsl_loader, self.args, session)
            
        return vl, va, true, pre

    def base_train(self, model, train_fsl_loader,train_gfsl_loader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()

        for _, batch in enumerate(zip(train_fsl_loader, train_gfsl_loader)):
             
            #划分支持集和查询集
            support_data, support_label = batch[0][0].cuda(), batch[0][1].cuda()
            query_data, query_label = batch[1][0].cuda(), batch[1][1].cuda()
            model.module.mode = 'classifier'
            # print(support_data.shape, query_data.shape, support_label.shape, epoch)
          
            logits = model(support_data, query_data, support_label,epoch)

            logits=logits[:,:args.base_class]
            # print(logits.shape, query_label.shape)
            # print(query_label)
            # print(query_label.view(-1, 1).repeat(1, args.num_tasks).view(-1).shape)
            total_loss = F.cross_entropy(logits, query_label.view(-1,1).repeat(1, args.num_tasks).view(-1).long())
            acc = count_acc(logits, query_label.view(-1,1).repeat(1, args.num_tasks).view(-1))

            lrc = scheduler.get_last_lr()[0]
            #tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} query acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_item=total_loss.item()
            
            del logits, total_loss  
        print('Session 0, epo {}, lrc={:.4f},total loss={:.4f} query acc={:.4f}'.format(epoch, lrc, total_loss_item, acc))
        print('Self.current_way:', model.module.current_way)
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    
    
    def test(self, model, testloader, train_fsl_loader,args, session,validation=True):
        true = []
        pre = []

     #计算当前会话中用于测试的类别数目
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()

     #初始化空张量 lgt 和 lbs，分别用于存储每次计算的预测值和真实标签
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                #将输入数据 data 输入到模型中，获取模型的输出 query
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
          
                logits = model.module.forward_many(query)
                
                logits = logits[:, :test_class]

                true.append(test_label)
                pre.append(logits)

                # 计算当前批次的交叉熵损失
                loss = F.cross_entropy(logits, test_label.long())
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            #将 Averager 对象中的累积值转换为标准的数值（item() 方法），得到最终的平均损失和平均准确率
            vl = vl.item()
            va = va.item()
        
        #将 lgt 张量的形状调整为 (batch_size * num_samples, test_class)，用于后续计算混淆矩
            lgt=lgt.view(-1, test_class)
            lbs=lbs.view(-1)

            
            if validation is not True:
                #构建保存混淆矩阵的路径
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
                
                #调用 confmatrix 函数生成混淆矩阵，生成的混淆矩阵将保存到 save_model_dir 路径
                cm=confmatrix(lgt,lbs,save_model_dir)

              #获取混淆矩阵的对角线元素，这代表每个类别的准确率
                perclassacc=cm.diagonal()
                #计算已见类别（base_class）的平均准确率
                seenac=np.mean(perclassacc[:args.base_class])
                #计算未见类别的平均准确率
                unseenac=np.mean(perclassacc[args.base_class:])

             #将已见类别和未见类别的准确率添加到 result_list 中，用于记录测试结果。   
                print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
                self.result_list.append('Seen Acc:%.5f, Unseen ACC:%.5f' % (seenac,unseenac))
                
                #self.analyze_logits(lgt,lbs,args,session)
        return vl, va, true, pre

   

    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        
        self.args.save_path = self.args.save_path + '%dSC-%dEpo-%.2fT-%dSshot' % (
            self.args.sample_class, self.args.epochs_base, self.args.temperature, self.args.sample_shot)
        
        self.args.save_path = self.args.save_path + '%.5fDec-%.2fMom-%dQ_' % (
            self.args.decay, self.args.momentum, self.args.batch_size_base,)
        

        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Lr1_%.6f-Lrg_%.5f-MS_%s-Gam_%.2f' % (
                self.args.lr_base, self.args.lrg, mile_stone, self.args.gamma,)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Lr1_%.6f-Lrg_%.5f-Step_%d-Gam_%.2f' % (
                self.args.lr_base, self.args.lrg, self.args.step, self.args.gamma,)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        # if self.args.debug:
        #     self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
