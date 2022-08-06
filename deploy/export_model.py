import argparse, paddle, os
from src.models import create_model

parser = argparse.ArgumentParser(description='PaddlePaddle TResNet ImageNet Inference')
parser.add_argument('--params_dir', type=str, default='params')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
args = parser.parse_args()

print('creating model...')
model = create_model(args)
model.eval()

model = paddle.jit.to_static(
    model,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None, 3, args.input_size, args.input_size], dtype='float32')
    ])

if args.input_size == 224:
    paddle.jit.save(model, os.path.join('JITMODEL/'+args.model_name, args.model_name))
elif args.input_size == 448:
    paddle.jit.save(model, os.path.join('JITMODEL/'+args.model_name+'_448', args.model_name+'_448'))
print('done\n')

