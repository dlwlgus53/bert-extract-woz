import os
import torch
import datetime, time
import argparse
from base_logger import logger
from dataset import Dataset
from utils import compute_F1, compute_exact_match
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from trainer import train, valid
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW
from knockknock import email_sender


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('--data_rate' ,  type = int, default=0.01)
parser.add_argument('--patience' ,  type = int, default=1)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained_model', type = str, default = 'bert-base-uncased', help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--dev_path' ,  type = str,  default = '../woz-data/MultiWOZ_2.1/dev_data.json')
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json')
parser.add_argument('--do_train' , default = True, help = 'do train or not', action=argparse.BooleanOptionalAction)
parser.add_argument('--num_worker',default=6, type=int,help='cpus')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=2, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
parser.add_argument('--max_length' , type = int,  default = 512, help = 'max length')

args = parser.parse_args()

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise


def main_worker(gpu, args):
    logger.info(f'{gpu} works!')
    batch_size = int(args.batch_size / args.gpus)
    num_worker = int(args.num_worker / args.gpus)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:3456',
        world_size=args.gpus,
        rank=gpu)
      
    torch.cuda.set_device(gpu)
    
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_trained_model).to(gpu)
    model = DDP(model, device_ids=[gpu])
    train_loader = DataLoader(args.train_dataset, batch_size, num_worker)
    dev_loader = DataLoader(args.val_dataset, batch_size, num_worker)
    
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    min_loss = float('inf')
    best_performance = {}

    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}


    if args.pretrained_model:
        logger.info("use trained model")
        model.load_state_dict(
            torch.load(args.pretrained_model, map_location=map_location))
    
    for epoch in range(args.max_epoch):
        dist.barrier()
        if gpu==0: logger.info(f"Epoch : {epoch}")
        if args.do_train:
            train(gpu, model, train_loader, optimizer) # TODO
        pred_texts, ans_texts, loss = valid(gpu, model, dev_loader, args.tokenizer)
        
        EM, F1 = 0, 0
        for iter, (pred_text, ans_text) in enumerate(zip(pred_texts, ans_texts)):
            EM += compute_exact_match(pred_text, ans_text)
            F1 += compute_F1(pred_text, ans_text)
        
        logger.info("Epoch : %d, EM : %.04f, F1 : %.04f, Loss : %.04f" % (epoch, EM/iter, F1/iter, loss))



        if loss < min_loss:
            print("New best")
            min_loss = loss
            best_performance['min_loss'] = min_loss.item()
            best_performance['EM'] = EM/iter
            best_performance['F1'] = F1/iter
            if not args.debugging:
                torch.save(model.state_dict(), f"model/woz{args.data_rate}.pt")
            logger.info("safely saved")
            
    logger.info(f"Best Score :  {best_performance}" )
    dist.barrier()

            
            

# @email_sender(recipient_emails=["jihyunlee@postech.ac.kr"], sender_email="knowing.deep.clean.water@gmail.com")
def main():
    makedirs("./data"); makedirs("./logs"); makedirs("./model");
    
    args.world_size = args.gpus * args.nodes 
    args.tokenizer = AutoTokenizer.from_pretrained(args.base_trained_model, use_fast=False)
    train_path = args.train_path[:-5] + str(args.data_rate) + '.json'
    args.train_dataset = Dataset(train_path, 'train', args.data_rate, args.tokenizer, False)
    args.val_dataset = Dataset(args.dev_path, 'dev', args.data_rate, args.tokenizer, False)
    
    mp.spawn(main_worker,
        nprocs=args.world_size,
        args=(args,),
        join=True)

    
if __name__ =="__main__":
    logger.info(f"{'-' * 30}")
    logger.info(args)
    logger.info("Start New Trainning")
    start = time.time()
    main()
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(result_list[0])
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")

    