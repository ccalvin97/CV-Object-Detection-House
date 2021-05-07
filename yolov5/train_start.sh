#!/bin/bash

## Start for Yolov5 ##

python train.py --img-size 1024 \
    --batch 6 \
    --epochs 50 \
    --data ./data/urbanisation_data_path.yaml \
    --cfg ./models/yolov5x.yaml \
    --weights yolov5x.pt \
    --device 0 \





: << !
parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path') - 模型配置文件，網路結�?    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path') - 數據集配置文件，資料路徑，類名等
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path') - 超参数文�?    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs') 
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes') - 输入图片分辨率大�?    parser.add_argument('--rect', action='store_true', help='rectangular training') - 是否采用矩形训练，默认False
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training') - 接着打断训练上次的结果接着训练
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint') - 不保存模型，默认False
    parser.add_argument('--notest', action='store_true', help='only test final epoch') - 不进行test，默认False
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check') - 不自动调整anchor，默认False
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters') - 是否进行超参数进化，默认False
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket') - 谷歌云盘bucket，一般不会用�?    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training') - 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training') - 加载的权重文�?    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') - 训练的设备，cpu�?(表示一个gpu设备cuda:0)�?,1,2,3(多个gpu设备)
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%') - 是否进行多尺度训练，默认False
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class') - 数据集是否只有一个类别，默认False
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer') - 是否使用adam优化�?    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode') - 是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify') - gpu编号
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers') -dataloader的最大worker数量
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    opt = parser.parse_args()

!
