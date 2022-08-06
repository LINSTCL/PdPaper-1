import paddle.vision
from paddle import inference
import numpy as np
from PIL import Image
import os

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

class InferenceEngine(object):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # init inference engine
        if args.input_size == 224:
            self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
                os.path.join(args.model_dir, f"{args.model_name}/{args.model_name}.pdmodel"),
                os.path.join(args.model_dir, f"{args.model_name}/{args.model_name}.pdiparams"))
        elif args.input_size == 448:
            self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
                os.path.join(args.model_dir, f"{args.model_name}_448/{args.model_name}_448.pdmodel"),
                os.path.join(args.model_dir, f"{args.model_name}_448/{args.model_name}_448.pdiparams"))
        # build transforms
        self.transforms = get_val_tfms(args)
        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.input_size, self.args.input_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        config.enable_memory_optim()
        if args.use_gpu:
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, x):
        x = x.flatten()
        class_id = x.argmax()
        prob = x[class_id]
        return class_id, prob

    def run(self, x):
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output

def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument('--model_name', type=str, default='tresnet_m')

    parser.add_argument("--model_dir", default='../JITMODEL', help="inference model dir")
    parser.add_argument("--use_gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--max-batch-size", default=16, type=int, help="max_batch_size")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")

    parser.add_argument('--val_zoom_factor', type=int, default=0.875)
    parser.add_argument("--input_size", default=224, type=int, help="crop_szie")
    parser.add_argument("--img-path", default="./images/demo.jpg")

    parser.add_argument("--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args

def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = inference_engine.preprocess(args.img_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    class_id, prob = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.img_path}, class_id: {class_id}, prob: {prob}")
    return class_id, prob


if __name__ == "__main__":
    args = get_args()
    class_id, prob = infer_main(args)
