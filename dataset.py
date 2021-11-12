import pdb
import json
import torch
import pickle
import ontology
from tqdm import tqdm
from base_logger import logger
from transformers import  AutoTokenizer
# here, squad means squad2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type, data_rate, tokenizer, debug=True):
        self.tokenizer = tokenizer
        self.encodings = []
        self.error_list = {}
        if type == 'train':
            raw_path = f'data/preprocessed_{type}_{data_rate}.pickle'
        else:
            raw_path = f'data/preprocessed_{type}.pickle'
            
        try:
            if debug:
                0/0
            else :    
                print(f"load {raw_path}")
                with open(raw_path, 'rb') as f:
                    encodings = pickle.load(f)
                    self.encodings = encodings
        except:
            print("preprocessing data...")
            raw_dataset = json.load(open(data_path, "r"))
            context, question, answer, dial_id, turn_id, schema = self._preprocessing_dataset(raw_dataset)
            assert len(context) == len(question) == len(answer['answer_start']) == len(answer['answer_end']) == len(dial_id) == len(turn_id) == len(schema)
            print("Encoding dataset (it will takes some time)")
            
            encodings = tokenizer(question, context, truncation='only_second', padding=True) # [CLS] question [SEP] context
            print("add token position")
            encodings = self._add_token_positions(encodings, answer)
            encodings.update({'dial_id' :dial_id, 'turn_id' : turn_id, 'schema' : schema})

            with open(raw_path, 'wb') as f:
                pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)

        self.encodings = encodings
        with open('./error.json','w') as f:
            json.dump(self.error_list,f, indent=4)
        

    def __getitem__(self, idx):
        to_be_tensor = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
        return {key: (torch.tensor(val[idx]) if key in to_be_tensor else val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    def _preprocessing_dataset(self, dataset):
        context = []
        question = []
        answer = {'answer_start' : [], 'answer_end' : []}
        dial_id = []
        turn_id = []
        schema = []
        
        print(f"preprocessing data")
        for id in dataset.keys():
            dialogue = dataset[id]['log']
            dialouge_text = ""
            for i, turn in enumerate(dialogue):
                dialouge_text += turn['user']
                
                for key in ontology.QA['extract-domain']:
                    q = ontology.QA[key]['description']
                    c = dialouge_text
                    
                    if len(c+q)+3 > self.tokenizer.model_max_length:
                        c = c[-(self.tokenizer.model_max_length-100):] # TODO
                        q = q[:100]

                    if key in turn['belief']: # 언급을 한 경우
                        a = turn['belief'][key]
                        if c.find(a) == -1:
                            self.error_list[id] = { 'context' : c, 'question' : q, 'answer' : a}
                            continue
                        else:
                            answer['answer_start'].append(c.find(a)) # 여기서 아마 문제가 생길걸?
                            answer['answer_end'].append(c.find(a) + len(a))
                                            
                    else:
                        answer['answer_start'].append(-1) 
                        answer['answer_end'].append(-1)
                    
                    schema.append(key)
                    context.append(c)
                    question.append(q)
                    dial_id.append(id)
                    turn_id.append(turn['turn_num'])
                     
                
                dialouge_text += turn['response']
        
        return context, question, answer, dial_id, turn_id,schema


    def _char_to_token_with_possible(self, i, encodings, char_position, type):
        if type == 'start':
            possible_position = [0,-1,1]
        else:
            possible_position = [-1,-2,0]

        for pp in possible_position:
            position = encodings.char_to_token(i, char_position + pp, sequence_index=1)
            if position != None:
                break
        return position

    def _add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers['answer_start'])):
            # char의 index로 되어있던것을 token의 index로 찾아준다.
            if  answers['answer_start'][i] != -1: # for case of mrq
                start_char = answers['answer_start'][i] 
                end_char = answers['answer_end'][i]
                start_position = self._char_to_token_with_possible(i, encodings, start_char,'start')
                end_position = self._char_to_token_with_possible(i, encodings, end_char,'end')
                start_positions.append(start_position)
                end_positions.append(end_position)
            else:
                start_positions.append(None)
                end_positions.append(None)
            
            if start_positions[-1] is None:
                # start_positions[-1] = self.tokenizer.model_max_length
                start_positions[-1] = 0
                
            if end_positions[-1] is None:
                # end_positions[-1] = self.tokenizer.model_max_length
                end_positions[-1] = 0
                
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

        

if __name__ == '__main__':
    data_path = '../data/MultiWOZ_2.1/dev_data.json'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    type = 'dev'
    dd = Dataset(data_path, type, tokenizer, True)
    pdb.set_trace()
    for i in range(10):
        print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])))

        # try:
        #     print(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])[dd[i]['start_positions']:dd[i]['end_positions']+1][0])
        # except:
        #     print(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])[dd[i]['start_positions']:dd[i]['end_positions']][0])
            
        


