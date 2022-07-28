import time, os
import paddle
import paddle.nn
import paddle.io
import paddle.vision
import numpy as np

def save_model(args, model:paddle.nn.Layer):
    temp = os.getcwd()
    os.chdir(args.params_path)
    paddle.save(model.state_dict(), args.model_name+'.pdparams')
    os.chdir(temp)

class my_DatasetFolder(paddle.vision.DatasetFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = paddle.to_tensor([target])
        return sample, target

def get_load_dataset(args):
    import os
    data_dir = ''
    if args.train_mode == True:
        data_dir = args.train_dir
    elif args.val_mode == True:
        data_dir = args.val_dir
    else:
        print('no dataset input')
        exit()
    datalist = os.listdir(data_dir)
    datalist.sort()
    i = 0
    for temp in datalist:
        if temp[0] == 'c':
            break
        os.rename(data_dir+'/'+temp, data_dir+'/class_'+str(i))
        i = i+1
    def val_tfms(input):
        input = np.array(input)
        input.resize([3,args.input_size,args.input_size])
        input = paddle.to_tensor(input, dtype=paddle.float32)
        return input
    dataset = my_DatasetFolder(data_dir, transform=val_tfms)
    batch_sampler = paddle.io.DistributedBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    train_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        return_list=True,
        use_shared_memory=False
    )
    return train_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape([-1]).cast('float32').sum(0, keepdim=True)
        res.append(correct_k.multiply(paddle.to_tensor([100.0 / batch_size], dtype=correct_k.dtype)))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self): self.reset()

    def reset(self): self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(model, val_loader):
    prec1_m = AverageMeter()
    last_idx = len(val_loader) - 1

    with paddle.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            output = model(input)

            prec1 = accuracy(output, target)
            prec1_m.update(prec1[0].item(), output.shape[0])

            if (last_batch or batch_idx % 100 == 0):
                log_name = 'ImageNet Test'
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Prec@1: {top1.val:>7.2f} ({top1.avg:>7.2f})  '.format(
                        log_name, batch_idx, last_idx,
                        top1=prec1_m))
    return prec1_m

def train(args, model):
    from paddle.distributed import fleet
    def optimizer_setting():
        import paddle.optimizer
        import paddle.regularizer
        optimizer = paddle.optimizer.SGD(
            learning_rate=args.learning_rate,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(args.l2_decay)
        )
        return optimizer
    fleet.init(is_collective=True)
    optimizer = optimizer_setting()
    optimizer = fleet.distributed_optimizer(optimizer)
    model = fleet.distributed_model(model)

    train_loader = get_load_dataset(args)

    for eop in range(args.epoch_num):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            out = model(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
            dy_out = avg_loss.numpy()
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
            if batch_id % 5 == 0:
                print(
                    "[Epoch %d, batch %d/%d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (
                        eop,
                        batch_id,
                        train_loader.dataset.__len__(),
                        dy_out,
                        acc_top1,
                        acc_top5
                    )
                )
        save_model(model)
    return
