import paddle
# from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse
from paddle.static import InputSpec
import numpy as np

# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir', type=str, default='val')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--remove_aa_jit', action='store_true', default=False)


def train(model, train_dataset):
    print('start training ... ')
    model.train()
    inverse_temperature = paddle.to_tensor(np.array([1.0/0.2], dtype='float32'))
    epoch_num = 20
    opt = paddle.optimizer.Adam(learning_rate=0.0001,
                                parameters=model.parameters())
    for epoch in range(epoch_num):
        for batch_id, [data, target] in enumerate(train_dataset):
            data = np.array(data, dtype='float32').transpose([2,0,1])
            data = np.array([data])
            cur_program = paddle.static.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="input",
                                    shape=data.shape,
                                    dtype='float32')
            loss = F.cross_entropy(similarities, sparse_labels)
            
            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()

def main():
    # parsing args
    args = parser.parse_args()

    # setup model
    print('creating model...')
    paddle.enable_static()
    input = InputSpec([1, 3, 224, 224], 'float32', 'x')
    label = InputSpec([1,1000], 'int64', 'label')
    model = paddle.Model(create_model(args), inputs=input, labels=label)
    print('creating model ok')
    # x = paddle.static.data(name="x", shape=[1, 3, 224, 224], dtype='float32')
    # z = model(x)
    # exe = paddle.static.Executor(paddle.CUDAPlace(0))
    # exe.run(paddle.static.default_startup_program())
    # prog = paddle.static.default_main_program()
    # paddle.save(prog.state_dict(), "TResNet.pdparams")
    # paddle.save(prog, "TResNet.pdmodel")

    # state = torch.load(args.model_path, map_location='cpu')['model']
    # model.load_state_dict(state, strict=False)
    # model.eval()
    # print('done\n')

    # setup data loader
    print('creating data loader...')
    val_loader = paddle.vision.DatasetFolder(args.val_dir)
    print('done\n')

    print('ready train...')
    model.prepare(
        optimizer=paddle.optimizer.Adam(learning_rate=0.01),
        loss=paddle.nn.MSELoss(),
        metrics=paddle.static.accuracy()
    )
    model.fit(
        val_loader,
        epochs=10,
        batch_size=48,
        verbose=1
    )

    # actual validation process
    # print('doing validation...')
    # prec1_f = validate(model, val_loader)
    # print("final top-1 validation accuracy: {:.2f}".format(prec1_f.avg))


if __name__ == '__main__':
    main()
