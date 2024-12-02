import numpy as np
import sys
import random
import torch
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from methods.fsaic import FSAIC
from methods.methods import run_method
import os 
from utils.utils import CL2N_embeddings,embedding_normalize,embs_norm_both
import sys

dataset_file = sys.argv[1]
merged_dict = np.load(dataset_file,allow_pickle=True)

out_dir = sys.argv[2]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

seed = 42

n_tasks = 100
batch_size = 5

args={}
args['iter']=30

use_mean = False
n_queries =[5]
k_shots = [3]
n_ways_effs = [1]

uniq_classes = sorted(list(set(merged_dict['concat_labels'])))

for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:
            print(f"Seed:{seed},Kshot:{k_shot},n_query:{n_query}")            
            
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # we set alpha value to the number of samples in query
            alpha = n_query
            alpha_glasso = 1000

            acc = {}
            acc["ss"] = []
            acc["smv"] = []
            acc["sscd"] = []

            acc["fsaic_centroid"] = []

            # Paddle methods evaluated
            acc["paddle"] = {}

            out_filename = f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}.json'
            out_file = os.path.join(out_dir,out_filename)
            
            task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                                n_tasks=n_tasks,
                                                n_ways=len(uniq_classes),
                                                n_ways_eff=n_ways_eff,
                                                n_query=n_query,
                                                k_shot=k_shot,
                                                seed=seed)

            test_embs, test_labels, test_audios, enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict) 
            # Normalize the extracted embeddings
            enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,use_mean=use_mean)
            
            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])
                
                eval = Simpleshot(avg="mean",backend="L2",method="ss")
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["ss"].extend(acc_list)
                
                if n_ways_eff == 1:
                    eval = Simpleshot(avg="mean",backend="L2",method="smv")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["smv"].extend(acc_list)

                    eval = FSAIC(method="centroid")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                    acc["fsaic_centroid"].extend(acc_list)
                    
                    eval = Simpleshot(avg="mean",backend="L2",method="sscd")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc['sscd'].extend(acc_list)
                
                else:
                    eval = Simpleshot(avg="mean",backend="cosine",method="ss")
                    acc_list,_,_ = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["ss"].extend(acc_list)

                args['maj_vote'] = False
                args['alpha'] = alpha
                method_info = {'device':'cpu','args':args}
                acc_list,_ = run_paddle_new(x_s, y_s, x_q, y_q,method_info,'paddle')                            
                acc['paddle'][str(alpha)].extend(acc_list)

                if n_ways_eff == 1:
                    args['maj_vote'] = True
                else:
                    args['maj_vote'] = False

                args['alpha'] = alpha
                method_info = {'device':'cuda','args':args}
                acc_list,_ = run_paddle_new(x_s, y_s, x_q, y_q,method_info,'paddle')                
                acc['paddle_maj'][str(alpha)].extend(acc_list)

                args['alpha'] = alpha_glasso
                method_info = {'device':'cuda','args':args}
                if n_ways_eff == 1:
                    try:
                        acc_list = run_2stage_paddle(x_s, y_s, x_q, y_q, test_audios[start:end], method_info)
                        acc['paddle_2stage'][str(alpha_glasso)].extend(acc_list)
                    except:
                        continue
            

            final_json = {}
            final_json['ss'] = 100*sum(acc["ss"])/len(acc["ss"])
            final_json['smv'] = 100*sum(acc["smv"])/len(acc["smv"])
            final_json['sscd'] = 100*sum(acc["sscd"])/len(acc["sscd"])
            final_json['fsaic_centroid'] = 100*sum(acc["fsaic_centroid"])/len(acc["fsaic_centroid"])
            final_json['paddle'] = {}
            final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])

            with open(out_file, 'w') as f:
                json.dump(final_json,f)

            with open(out_file,'w') as f:
                json.dump(final_json,f)
