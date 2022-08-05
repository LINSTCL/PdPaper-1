import time, os
import paddle
import paddle.nn
import paddle.io
import paddle.vision
from paddle.distributed import fleet
import PIL.Image

def save_model(args, model:paddle.nn.Layer, epoch_num):
    paddle.save(model.state_dict(), args.params_dir+'/'+str(epoch_num)+'-'+args.model_name+'.pdparams')

class my_DatasetFolder(paddle.vision.DatasetFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = paddle.to_tensor([target])
        return sample, target

def get_val_tfms(args):
    if args.input_size == 448: # squish
        val_tfms = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Resize((args.input_size, args.input_size))])
    else: # crop
        val_tfms = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Resize(int(args.input_size / args.val_zoom_factor)),
            paddle.vision.transforms.CenterCrop(args.input_size)])
    val_tfms.transforms.append(paddle.vision.transforms.ToTensor())
    return val_tfms

def get_load_dataset(args):
    data_dir = args.data_dir
    val_tfms = get_val_tfms(args)
    dataset = my_DatasetFolder(data_dir, transform=val_tfms)
    batch_sampler = paddle.io.DistributedBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    train_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        return_list=True,
        use_shared_memory=False
    )
    return train_loader

class forecastDataset(paddle.io.Dataset):
    def __init__(self, args, transform=None):
        super(forecastDataset, self).__init__()
        self.transform = transform
        self.extensions = (
            '.jpg', '.jpeg', '.png', '.ppm',
            '.bmp', '.pgm', '.tif', '.tiff',
            '.webp'
        )
        file_list = os.listdir(args.data_dir)
        self.file_list = []
        for temp in file_list:
            if self.is_valid_file(args.data_dir+'/'+temp):
                self.file_list.append(args.data_dir+'/'+temp)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.file_list[idx])
        img = self.transform(img)
        if type(img) == paddle.Tensor:
            if len(img.shape) == 3:
                img = paddle.to_tensor([img])
        return img, self.file_list[idx]

    def __len__(self):
        return len(self.file_list)

    def has_valid_extension(self, filename, extensions):
        assert isinstance(extensions, (list, tuple)), ("`extensions` must be list or tuple.")
        extensions = tuple([x.lower() for x in extensions])
        return filename.lower().endswith(extensions)

    def is_valid_file(self, x):
        return self.has_valid_extension(x, self.extensions)

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

            log_name = 'ImageNet Test'
            print(
                '{0}: [{1:>4d}/{2}]  '
                'Prec@1: {top1.val:>7.2f} ({top1.avg:>7.2f})  '.format(
                    log_name, batch_idx, last_idx,
                    top1=prec1_m))

    return prec1_m

def train(args, model):
    def optimizer_setting():
        import paddle.optimizer
        import paddle.regularizer
        optimizer = paddle.optimizer.SGD(
            learning_rate=args.lr,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(args.l2_decay)
        )
        return optimizer
    optimizer = optimizer_setting()
    optimizer = fleet.distributed_optimizer(optimizer)
    model = fleet.distributed_model(model)

    train_loader = get_load_dataset(args)
    t1 = time.time()
    t2 = time.time()

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
            t2 = time.time()
            if batch_id % 10 == 0:
                print(
                    "[Epoch %d, batch %d/%d, time %.2f] loss: %.5f, acc1: %.5f, acc5: %.5f" % (
                        eop,
                        batch_id*args.batch_size,
                        train_loader.dataset.__len__(),
                        t2-t1,
                        dy_out,
                        acc_top1,
                        acc_top5
                    )
                )
                t1 = time.time()
        save_model(args, model, eop)
    return
