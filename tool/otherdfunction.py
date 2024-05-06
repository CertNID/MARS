


 
def get_architecture(arc:str, dataset: str) -> torch.nn.Module:
    
    if dataset == "ImageNet" and arc == "resnet50":
        model = torch.nn.DataParallel(resnet50(pretrained = True)).to(device)
        cudnn.benchmark = True
        normalize_layer = get_normalize_layer(dataset)                  
        return torch.nn.Sequential(normalize_layer, model)
    elif (dataset == "ids18" or dataset == "ids18_unnormalized")and arc == "acid":
        return AdaptiveClustering(get_dataset(args.dataset, "train")[0].shape[1], get_num_classes(args.dataset), device)
    elif (dataset == 'newhulk' or dataset == 'newinfiltration') and arc == 'cade':
        train, _, num_classes_train, _ = get_dataset(dataset, "train")
        return Autoeconder(train.size(1), num_classes_train, device)
            

def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    ##=========================Maggie=========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ##========================================================
    
    batch_time = Tracker()
    data_time = Tracker()
    losses = Tracker()
    top1 = Tracker()
    top5 = Tracker()
    end = time.time()
    model.train()                                     

    for i, (inputs, targets) in enumerate(loader):      
        
        data_time.update(time.time() - end)


        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs + torch.randn_like(inputs, device=device) * noise_sd
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)

def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    ##=========================Maggie=========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ##========================================================
        
    batch_time = Tracker()
    data_time = Tracker()
    losses = Tracker()
    top1 = Tracker()
    end = time.time()

    
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
        
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            
            acc1= accuracy(outputs, targets, topk=1)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))

            
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {top1.val:.3f} ({top1.avg:.3f})'
                    .format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

        return (losses.avg, top1.avg)
   


    # pin_memory = 0
    # if (args.dataset == "ImageNet" and args.model == "res50"):
    #     pin_memory = (args.dataset == "ImageNet")          
    # elif ((args.dataset == "ids18" or args.dataset == "ids18_unnormalized")and args.model == 'acid') :
    #     pin_memory = (args.dataset == 'ids18')
    # elif((args.dataset == 'newhulk' or "newinfiltration")and args.model == 'cade'):
    #     pin_memory = (args.dataset == args.dataset)
    
    # x_train, y_train,_,_ = get_dataset(args.dataset, "train")
    # nonbindex = np.where(y_train == 0)[0]
    # nonbindex1 = np.where(y_train == 1)[0]
    # nonbindex2 = np.where(y_train == 2)[0]
    # x_train0 = x_train[nonbindex].numpy()
    # x_train2 = x_train[nonbindex1].numpy()
    # x_train3 = x_train[nonbindex2].numpy()
    # print(x_train0.max(),x_train0.min(),x_train2.max(),x_train2.min(),x_train3.max(),x_train3.min())
    # print(x_train0.shape,x_train2.shape,x_train3.shape)
    # x_train1, y_train1,_,_,_ = get_dataset(args.dataset, "test")
    # nonbindex = np.where(y_train1 == 0)[0]
    # nonbindex1 = np.where(y_train1 == 1)[0]
    # nonbindex2 = np.where(y_train1 == 2)[0]
    # nonbindex3 = np.where(y_train1 == 3)[0]
    # x_train0 = x_train1[nonbindex].numpy()
    # x_train2 = x_train1[nonbindex1].numpy()
    # x_train3 = x_train1[nonbindex2].numpy()
    # x_train4 = x_train1[nonbindex3].numpy()
    # print(x_train4.shape)
    # print(x_train0.max(),x_train0.min(),x_train2.max(),x_train2.min(),x_train3.max(),x_train3.min(),x_train4.max(),x_train4.min())


