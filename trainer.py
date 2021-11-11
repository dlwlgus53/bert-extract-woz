
import torch
from tqdm import tqdm
from base_logger import logger


def train(gpu, model, train_loader, optimizer):
    model.train()
    loss_sum = 0
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].cuda(non_blocking = True)
        attention_mask = batch['attention_mask'].cuda(non_blocking = True)
        start_positions = batch['start_positions'].cuda(non_blocking = True)
        end_positions = batch['end_positions'].cuda(non_blocking = True)
        
        outputs = model(input_ids = input_ids, attention_mask= attention_mask,\
            start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        if (iter + 1) % 10 == 0 and gpu==0:
            logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
            gpu,
            iter, 
            str(len(train_loader)),
            loss.detach())
            )
                

def valid(gpu, model, dev_loader, tokenizer):
    model.eval()
    pred_texts = []
    ans_texts = []
    loss_sum = 0
    print("Validation start")
    with torch.no_grad():
        for iter,batch in enumerate(dev_loader):
            input_ids = batch['input_ids'].cuda(non_blocking = True)
            attention_mask = batch['attention_mask'].cuda(non_blocking = True)
            start_positions = batch['start_positions'].cuda(non_blocking = True)
            end_positions = batch['end_positions'].cuda(non_blocking = True)
            
            outputs = model(input_ids = input_ids, attention_mask= attention_mask,\
                start_positions=start_positions, end_positions=end_positions)
            pred_start_positions = torch.argmax(outputs['start_logits'], dim=1).to('cpu')
            pred_end_positions = torch.argmax(outputs['end_logits'], dim=1).to('cpu')
            
            
            for b in range(len(pred_start_positions)):
                ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                ans_texts.append(ans_text)
                pred_texts.append(pred_text)

   
            loss = outputs[0].to('cpu')
            loss_sum += loss
            if (iter + 1) % 10 == 0 and gpu==0:
                logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
                gpu,
                iter, 
                str(len(dev_loader)),
                loss.detach())
            )
                    
    
    return pred_texts, ans_texts, loss_sum/iter
        