import paddle
from paddle.distributed import fleet
# from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse
import os
from visualdl import LogWriter

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--params_path', type=str, default='params')
parser.add_argument('--train_dir', type=str, default='train')
parser.add_argument('--val_dir', type=str, default='val')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--train_mode', action='store_true', default=False)

parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--momentum_rate', type=float, default=0.9)
parser.add_argument('--l2_decay', type=float, default=1e-4)

parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--epoch_num', type=int, default=10)

def save_model(args):
    prog = paddle.static.default_main_program()
    paddle.save(prog.state_dict(), args.params_path+'/'+args.model_name+'.pdparams')
    print('Model parameters saved')

class my_DatasetFolder(paddle.vision.DatasetFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = paddle.to_tensor([target])
        return sample, target

def get_load_dataset(args, feed_list, place):
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
    if args.input_size == 448: # squish
        val_tfms = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Resize((args.input_size, args.input_size))])
    else: # crop
        val_tfms = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Resize(int(args.input_size / args.val_zoom_factor)),
             paddle.vision.transforms.CenterCrop(args.input_size)])
    val_tfms.transforms.append(paddle.vision.transforms.ToTensor())
    dataset = my_DatasetFolder(data_dir, transform=val_tfms)
    train_loader = paddle.io.DataLoader(
        dataset,
        return_list=False,
        feed_list=feed_list,
        batch_size=args.batch_size,
        num_workers=1
    )
    return train_loader

def optimizer_setting(args, parameter_list=None):
    boundaries = [20, 40, 60, 80, 100]
    values = [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries= boundaries,
        values=values,
        verbose=True
    )
    optimizer = paddle.optimizer.SGD(
        learning_rate=learning_rate
    )
    # optimizer = paddle.fluid.optimizer.Adam(learning_rate=args.base_lr)
    # optimizer = paddle.optimizer.Momentum(
    #     learning_rate=args.base_lr,
    #     momentum=args.momentum_rate,
    #     weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
    #     parameters=parameter_list)
    return optimizer, learning_rate

def train(args, model):
    print('start training ... ')
    image = paddle.static.data(name="image", shape=[args.batch_size, 3, 224, 224], dtype='float32')
    label= paddle.static.data(name="label", shape=[args.batch_size, 1], dtype='int64')
    # out = model.net(image)
    out = model(image)

    # avg_cost = paddle.nn.functional.dice_loss(input=out, label=label)
    avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)

    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    print("Run on {}.".format(place))
    print('creating data loader...')
    data_loader = get_load_dataset(args, [image, label], place)
    print('done\n')
    # for temp in data_loader:
    #     print(temp[0]['y'])
    #     exit()

    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)
    optimizer, learning_rate = optimizer_setting(args)

    optimizer = fleet.distributed_optimizer(optimizer)
    optimizer.minimize(avg_cost)

    exe = paddle.static.Executor(place)
    print("Execute startup program.")
    exe.run(paddle.static.default_startup_program())
    print('done\n')
    # prog = paddle.static.default_main_program()
    # paddle.save(prog.state_dict(), args.params_path+'/'+args.model_name+'.pdparams')
    # return

    try:
        print('load params')
        if '.pdparams' in args.model_name:
            print('--model_name arg Do not add \".pdparams\"')
            return
        prog = paddle.static.default_main_program()
        state_dict = paddle.load(args.params_path+'/'+args.model_name+'.pdparams')
        prog.set_state_dict(state_dict)
        print('done\n')
    except:
        print('create new params')
        prog = paddle.static.default_main_program()
        paddle.save(prog.state_dict(), args.params_path+'/'+args.model_name+'.pdparams')
        print('done\n')

    writer = LogWriter(logdir="log/train")
    step = 1

    for eop in range(args.epoch_num):
        for batch_id, data in enumerate(data_loader()):
            loss, acc1 = exe.run(
                paddle.static.default_main_program(),
                feed=data,
                fetch_list=[avg_cost.name, acc_top1.name]
            )
            writer.add_scalar(tag="loss", step=step, value=loss)
            step = step+1
            if batch_id % 10 == 0:
                print(
                        "[Epoch %d, batch %d/%d] loss: %.5f, acc1: %.5f" % (
                            eop,
                            batch_id*args.batch_size,
                            data_loader.dataset.__len__(),
                            loss,
                            acc1
                        )
                    )
        save_model(args)
        learning_rate.step()

def main():
    # parsing args
    args = parser.parse_args()
    # setup model
    print('creating model...')
    paddle.enable_static()
    model = paddle.vision.ResNet(paddle.vision.models.resnet.BasicBlock, 50)
    # model = create_model(args)
    print('done\n')

    if args.train_mode == True:
        train(args, model)

if __name__ == '__main__':
    main()
