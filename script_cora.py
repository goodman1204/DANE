import numpy as np
import linecache
from Dataset.dataset import Dataset
from Model.model import Model
from Trainer.trainer import Trainer
from Trainer.pretrainer import PreTrainer
from Utils import gpu_info
import os
import random

from utils import preprocess_graph, get_roc_score, sparse_to_tuple,sparse_mx_to_torch_sparse_tensor,cluster_acc,clustering_evaluation, find_motif,drop_feature, drop_edge,choose_cluster_votes,plot_tsne,save_results,entropy_metric,plot_tsne_non_centers
from evaluation import clustering_latent_space
from hungrian import label_mapping
from collections import Counter
import argparse
import time


if __name__=='__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="Node clustering")
        parser.add_argument('--model', type=str, default='DANE', help="models used for clustering: gcn_ae,gcn_vae,gcn_vaecd")
        parser.add_argument('--seed', type=int, default=20, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
        parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
        parser.add_argument('--hidden2', type=int, default=200, help='Number of units in hidden layer 2.')
        parser.add_argument('--lr', type=float, default=0.002, help='Initial aearning rate.')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
        parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')

        parser.add_argument('--synthetic_num_nodes',type=int,default=1000)
        parser.add_argument('--synthetic_density', type=float, default=0.1)

        parser.add_argument('--nClusters',type=int,default=7)
        parser.add_argument('--num_run',type=int,default=5,help='Number of running times')
        parser.add_argument('--cuda', type=int, default=0, help='training with GPU.')
        args, unknown = parser.parse_known_args()

        return args
    args = parse_args()

    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print(gpus_to_use, free_memory)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

    # random.seed(9001)

    dataset_config = {'feature_file': './Database/{}/features.txt'.format(args.dataset),
                      'graph_file': './Database/{}/edges.txt'.format(args.dataset),
                      'walks_file': './Database/{}/walks.txt'.format(args.dataset),
                      'label_file': './Database/{}/group.txt'.format(args.dataset)}
    graph = Dataset(dataset_config)

    pretrain_config = {
        'net_shape': [200, 100],
        'att_shape': [200, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'pretrain_params_path': './Log/{}/pretrain_params.pkl'.format(args.dataset)}

    model_config = {
        'net_shape': [200, 100],
        'att_shape': [200, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/{}/pretrain_params.pkl'.format(args.dataset)
    }

    trainer_config = {
        'net_shape': [200, 100],
        'att_shape': [200, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'drop_prob': 0.2,
        'learning_rate': 1e-5,
        'batch_size': 100,
        'num_epochs': 500,
        'beta': 100,
        'alpha': 50,
        'gamma': 500,
        'model_path': './Log/{}/{}_model.pkl'.format(args.dataset,args.dataset),
    }

    start_time = time.time()

    pretrainer = PreTrainer(pretrain_config)
    pretrainer.pretrain(graph.X, 'net')
    pretrainer.pretrain(graph.Z, 'att')

    model = Model(model_config)
    trainer = Trainer(model, trainer_config)
    trainer.train(graph)
    # trainer.infer(graph)
    Z,Y = trainer.get_embedding(graph)
    print("Z shape:",Z.shape)
    print("Y shape:",Y.shape)
    print("raw Z shape:",graph.Z.shape)
    print("Y shape:",graph.Y.shape)

    z = Z
    Y = Y.argmax(1)

    end_time = time.time()


    mean_h=[]
    mean_c=[]
    mean_v=[]
    mean_ari=[]
    mean_ami=[]
    mean_nmi=[]
    mean_purity=[]
    mean_accuracy=[]
    mean_f1=[]
    mean_precision=[]
    mean_recall = []
    mean_entropy = []
    mean_time= []



    tru=Y


    pre,mu_c=clustering_latent_space(z,tru)

    plot_tsne_non_centers(args.dataset,args.model,args.epochs,z,Y,pre)

    for i in range(args.num_run):
        pre,mu_c=clustering_latent_space(z,tru)


        pre = label_mapping(tru,pre)
        H, C, V, ari, ami, nmi, purity,f1_score,precision,recall= clustering_evaluation(tru,pre)

        entropy = entropy_metric(tru,pre)

        acc = cluster_acc(pre,tru)[0]
        mean_h.append(round(H,4))
        mean_c.append(round(C,4))
        mean_v.append(round(V,4))
        mean_ari.append(round(ari,4))
        mean_ami.append(round(ami,4))
        mean_nmi.append(round(nmi,4))
        mean_purity.append(round(purity,4))
        mean_accuracy.append(round(acc,4))
        mean_f1.append(round(f1_score,4))
        mean_precision.append(round(precision,4))
        mean_recall.append(round(recall,4))
        mean_entropy.append(round(entropy,4))
        mean_time.append(round(end_time-start_time,4))

    # metrics_list=[mean_h,mean_c,mean_v,mean_ari,mean_ami,mean_nmi,mean_purity,mean_accuracy,mean_f1,mean_precision,mean_recall,mean_entropy]
    metrics_list=[mean_h,mean_c,mean_v,mean_ari,mean_ami,mean_nmi,mean_purity,mean_accuracy,mean_f1,mean_precision,mean_recall,mean_entropy,mean_time]
    save_results(args,metrics_list)

    ###### Report Final Results ######
    print('Homogeneity:{}\t mean:{}\t std:{}\n'.format(mean_h,round(np.mean(mean_h),4),round(np.std(mean_h),4)))
    print('Completeness:{}\t mean:{}\t std:{}\n'.format(mean_c,round(np.mean(mean_c),4),round(np.std(mean_c),4)))
    print('V_measure_score:{}\t mean:{}\t std:{}\n'.format(mean_v,round(np.mean(mean_v),4),round(np.std(mean_v),4)))
    print('adjusted Rand Score:{}\t mean:{}\t std:{}\n'.format(mean_ari,round(np.mean(mean_ari),4),round(np.std(mean_ari),4)))
    print('adjusted Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_ami,round(np.mean(mean_ami),4),round(np.std(mean_ami),4)))
    print('Normalized Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_nmi,round(np.mean(mean_nmi),4),round(np.std(mean_nmi),4)))
    print('Purity:{}\t mean:{}\t std:{}\n'.format(mean_purity,round(np.mean(mean_purity),4),round(np.std(mean_purity),4)))
    print('Accuracy:{}\t mean:{}\t std:{}\n'.format(mean_accuracy,round(np.mean(mean_accuracy),4),round(np.std(mean_accuracy),4)))
    print('F1-score:{}\t mean:{}\t std:{}\n'.format(mean_f1,round(np.mean(mean_f1),4),round(np.std(mean_f1),4)))
    print('precision_score:{}\t mean:{}\t std:{}\n'.format(mean_precision,round(np.mean(mean_precision),4),round(np.std(mean_precision),4)))
    print('recall_score:{}\t mean:{}\t std:{}\n'.format(mean_recall,round(np.mean(mean_recall),4),round(np.std(mean_recall),4)))
    print('entropy:{}\t mean:{}\t std:{}\n'.format(mean_entropy,round(np.mean(mean_entropy),4),round(np.std(mean_entropy),4)))
    print("True label distribution:{}".format(tru))
    print(Counter(tru))
    print("Predicted label distribution:{}".format(pre))
    print(Counter(pre))
