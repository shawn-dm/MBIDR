from datetime import time

import shutil
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import copy
from utility.helper import *
from utility.batch_test import *
import multiprocessing
import torch.multiprocessing
import random
from Gumbel_Softmax import Gating_Net

class MBIDR(nn.Module):
    name = 'MBIDR'

    def __init__(self, max_item_list, data_config, args):
        super(MBIDR, self).__init__()
        # ********************** input data *********************** #
        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj).to(device) for adj in self.pre_adjs]
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        # ********************** hyper parameters *********************** #
        self.coefficient = torch.tensor(eval(args.coefficient)).view(1, -1).to(device)
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)  # dropout ratio
        self.nhead = args.nhead
        self.att_dim = args.att_dim
        self.temp = args.temp
        # ********************** learnable parameters *********************** #
        self.all_weights = {}
        self.all_weights['user_embedding'] = Parameter(torch.FloatTensor(self.n_users, self.emb_dim))
        self.all_weights['item_embedding'] = Parameter(torch.FloatTensor(self.n_items, self.emb_dim))
        self.all_weights['relation_embedding'] = Parameter(torch.FloatTensor(self.n_relations, self.emb_dim))

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            self.all_weights['W_gc_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.all_weights['W_rel_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))

        self.all_weights['trans_weights_s1'] = Parameter(
            torch.FloatTensor(self.n_relations, self.emb_dim, self.att_dim))
        self.all_weights['trans_weights_s2'] = Parameter(torch.FloatTensor(self.n_relations, self.att_dim, 1))
        self.reset_parameters()
        self.all_weights = nn.ParameterDict(self.all_weights)
        self.dropout = nn.Dropout(self.mess_dropout[0], inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.all_weights['user_embedding'])
        nn.init.xavier_uniform_(self.all_weights['item_embedding'])
        nn.init.xavier_uniform_(self.all_weights['relation_embedding'])
        nn.init.xavier_uniform_(self.all_weights['trans_weights_s1'])
        nn.init.xavier_uniform_(self.all_weights['trans_weights_s2'])
        for k in range(self.n_layers):
            nn.init.xavier_uniform_(self.all_weights['W_gc_%d' % k])
            nn.init.xavier_uniform_(self.all_weights['W_rel_%d' % k])

    def _convert_sp_mat_to_sp_tensor(self, X):
        '''print(f"X的类型:\n{type(X)}") # <class 'scipy.sparse.csr.csr_matrix'>'''
        coo = X.tocoo()
        values = coo.data
        # 创建一个二维数组，其中第一行是行索引，第二行是列索引。
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    def disentangle_adj(self, X, X_csr, gumb_out):

        X_coo = X_csr.tocoo()
        shape = X_coo.shape
        # 获取原始的索引和值
        rows, cols = X.indices()
        values = X.values()
        # 从X中筛选与X_dense的前n_users行对应的非零元素的索引
        mask_users = rows < n_users
        # 对应的gumb_out值
        mask_zero = gumb_out[:, 0].bool()
        # 为X1和X2构建索引和值
        filtered_rows = rows[mask_users]
        filtered_cols = cols[mask_users]

        X1_rows = torch.cat([filtered_rows[~mask_zero], filtered_cols[~mask_zero]])
        X1_cols = torch.cat([filtered_cols[~mask_zero], filtered_rows[~mask_zero]])
        X1_values = torch.cat([values[mask_users][~mask_zero], values[mask_users][~mask_zero]])

        X2_rows = torch.cat([filtered_rows[mask_zero], filtered_cols[mask_zero]])
        X2_cols = torch.cat([filtered_cols[mask_zero], filtered_rows[mask_zero]])
        X2_values = torch.cat([values[mask_users][mask_zero], values[mask_users][mask_zero]])

       
        # 转换回稀疏张量格式
        X1 = torch.sparse.FloatTensor(torch.stack([X1_rows, X1_cols]), X1_values, torch.Size(shape)).to(device).coalesce()
        X2 = torch.sparse.FloatTensor(torch.stack([X2_rows, X2_cols]), X2_values, torch.Size(shape)).to(device).coalesce()

        return X1, X2

    # def forward(self, sub_mats, device):
    def forward(self, device):

        ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']),
                                   dim=0).unsqueeze(1).repeat(1, self.n_relations, 1)
        
        all_embeddings = ego_embeddings
       
        all_rela_embs = {}
        for i in range(self.n_relations):
            beh = self.behs[i] # behs = ['pv', 'cart', 'train']
            rela_emb = self.all_weights['relation_embedding'][i]
            rela_emb = torch.reshape(rela_emb, (-1, self.emb_dim))
            all_rela_embs[beh] = [rela_emb]

        total_mm_time = 0.
        # 对网络的每一层进行操作
        for k in range(0, self.n_layers):
            embeddings_list = []
            # 对于每个行为子图，计算其节点嵌入
            for i in range(self.n_relations):
                st = time()
                
                '''根据拉普拉斯矩阵中前n_user行非零元素的位置，将对应的用户嵌入和物品嵌入沿第1维连接在一起'''
                # 确保稀疏张量是合并的
                self.pre_adjs_tensor[i] = self.pre_adjs_tensor[i].coalesce()
                # 获取self.pre_adjs_tensor[i]的索引
                rows = self.pre_adjs_tensor[i].indices()[0]
                cols = self.pre_adjs_tensor[i].indices()[1]
                # 筛选出前n_users行中的非零元素的索引
                selected_indices = cols[rows < n_users]
                # 对每一对索引从ego_embeddings[:, i, :]中选择对应的嵌入，并沿第1维连接
                cat_embeddings = torch.cat([ego_embeddings[:, i, :][rows[rows < n_users]], ego_embeddings[:, i, :][selected_indices]], dim=1)

                '''利用Gating_Net判断交互关系所属类别'''
                gating_model_adj = Gating_Net(self.emb_dim, 2, [self.emb_dim, 2]).to(device)
                gumb_out_adj = gating_model_adj(cat_embeddings, self.temp, True, 2)

                '''根据交互关系类别对邻接矩阵（拉普拉斯矩阵）解耦'''
                sub_adj_1, sub_adj_2 = self.disentangle_adj(self.pre_adjs_tensor[i], self.pre_adjs[i], gumb_out_adj)

                '''在两个意图子图上进行图卷积'''
                embeddings1_ = torch.matmul(sub_adj_1, ego_embeddings[:, i, :])
                embeddings2_ = torch.matmul(sub_adj_2, ego_embeddings[:, i, :])
                total_mm_time += time() - st
                rela_emb = all_rela_embs[self.behs[i]][k]
                # 节点嵌入传播聚合，对应论文公式(1)
                embeddings1_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings1_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings2_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings2_, rela_emb), self.all_weights['W_gc_%d' % k]))

                embeddings_list.append(embeddings1_)
                embeddings_list.append(embeddings2_)
            # 最终的embeddings_list=[[embeddings1_r1],[embeddings2_r1],...,[embeddings1_rn],[embeddings2_rn]]

            # 将每个节点在所有行为下（每个行为包含两个意图子图）的嵌入连接起来，用于后续的相关性计算
            embeddings_st = torch.stack(embeddings_list, dim=1)
            

            embeddings_list = []
            attention_list = []
            # 对于每个行为（的每个意图子图），计算其注意力权重
            for i in range(self.n_relations):
                attention = F.softmax( 
                    torch.matmul( # trans_weights_s1[i].shape = (emb_dim, att_dim)
                        torch.tanh(torch.matmul(embeddings_st, self.all_weights['trans_weights_s1'][i])), # (n_node, 2*n_rel, att_dim)
                        self.all_weights['trans_weights_s2'][i] # (n_node, 2*n_rel, 1)
                    ).squeeze(2),
                    dim=1
                ).unsqueeze(1) # (n_node, 1, 2*n_rel)
                # print("attention的形状：",attention.shape)
                
            
                attention_list.append(attention)
                # 根据注意力权重对嵌入张量加权更新
                embs_cur_rela = torch.matmul(attention, embeddings_st).squeeze(1)
                # (n_node, 1, n_rel) * (n_node, n_rel, emb_dim) = (n_node, 1, emb_dim)
                embeddings_list.append(embs_cur_rela) # 列表中共n_rel个元素 
               
            ego_embeddings = torch.stack(embeddings_list, dim=1)
            # ego_embeddings = embs_cur_rela.repeat(1, self.n_relations, 1)
            attn = torch.cat(attention_list, dim=1)
            ego_embeddings = self.dropout(ego_embeddings)
            all_embeddings = all_embeddings + ego_embeddings # 将所有层嵌入连接在一起

            # 更新边嵌入
            for i in range(self.n_relations):
                rela_emb = torch.matmul(all_rela_embs[self.behs[i]][k],
                                        self.all_weights['W_rel_%d' % k])
                all_rela_embs[self.behs[i]].append(rela_emb)

        all_embeddings /= self.n_layers + 1
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        token_embedding = torch.zeros([1, self.n_relations, self.emb_dim], device=device)
        i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)

        attn_user, attn_item = torch.split(attn, [self.n_users, self.n_items], 0)

        for i in range(self.n_relations):
            all_rela_embs[self.behs[i]] = torch.mean(torch.stack(all_rela_embs[self.behs[i]], 0), 0)


        return u_g_embeddings, i_g_embeddings, all_rela_embs, attn_user, attn_item


'''非采样损失'''
class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, rela_embeddings):
        uid = ua_embeddings[input_u]
        uid = torch.reshape(uid, (-1, self.n_relations, self.emb_dim))
        pos_r_list = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            pos_beh = ia_embeddings[:, i, :][label_phs[i]]  # [B, max_item, dim]
            pos_num_beh = torch.ne(label_phs[i], self.n_items).float()
            pos_beh = torch.einsum('ab,abc->abc', pos_num_beh,
                                   pos_beh)  # [B, max_item] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ac,abc->abc', uid[:, i, :],
                                 pos_beh)  # [B, dim] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ajk,lk->aj', pos_r, rela_embeddings[beh])
            pos_r_list.append(pos_r)

        loss = 0.
        for i in range(self.n_relations):
            beh = self.behs[i]
            temp = torch.einsum('ab,ac->bc', ia_embeddings[:, i, :], ia_embeddings[:, i, :]) \
                   * torch.einsum('ab,ac->bc', uid[:, i, :], uid[:, i, :])  # [B, dim]' * [B, dim] -> [dim, dim]
            tmp_loss = self.wid[i] * torch.sum(
                temp * torch.matmul(rela_embeddings[beh].T, rela_embeddings[beh]))
            tmp_loss += torch.sum((1.0 - self.wid[i]) * torch.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss += self.coefficient[i] * tmp_loss

        regularizer = torch.sum(torch.square(uid)) * 0.5 + torch.sum(torch.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss, emb_loss

def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_behs)]  #

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]  # ndarray [x,]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])


def test_torch(ua_embeddings, ia_embeddings, rela_embedding, users_to_test, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, rela_embedding, users, items):
        ug_embeddings = ua_embeddings[users]  # []
        pos_ig_embeddings = ia_embeddings[items]
        dot = np.multiply(pos_ig_embeddings, rela_embedding)  # [I, dim] * [1, dim]-> [I, dim]
        batch_ratings = np.matmul(ug_embeddings, dot.T)  # [U, dim] * [dim, I] -> [U, I]
        return batch_ratings

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    test_users = users_to_test
    n_test_users = len(test_users)

    # pool = torch.multiprocessing.Pool(cores)
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, rela_embedding, user_batch, item_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users

    pool.close()
    return result

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4090'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    # 设置pytorch的自动求导机制在计算中捕获潜在的异常
    torch.autograd.set_detect_anomaly(True)
    # 设置GIT PYTHON库的行为，将刷新设置为quiet，避免产生过多的GIT库相关的输出
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    # 设置CUDA的启动模式为阻塞模式，这将使得所有CUDA函数的调用都变成同步执行，可以方便地进行调试，但可能会导致性能下降
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # set_seed(2020)
    set_seed(2020)

    config = dict()
    config['device'] = device
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['behs'] = data_generator.behs
    config['trn_mat'] = data_generator.trnMats[-1]  # 目标行为交互矩阵

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    生成拉普拉斯矩阵，其中每个条目定义了两个连接节点之间的衰减因子
    """


    # 使用数据生成器的get_adj_mat方法计算邻接矩阵
    pre_adj_list = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list
    print('use the pre adjcency matrix')
    # 从数据生成器中获取用户数和物品数
    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    # 获取行为数量
    n_behs = data_generator.beh_num

    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []
    beh_label_list = []
    for i in range(n_behs):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time()

    model = MBIDR(max_item_list, data_config=config, args=args).to(device)
    # augmentor = Augmentor(data_config=config, args=args)
    recloss = RecLoss(data_config=config, args=args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # hmg = HMG(model.parameters(), relax_factor=args.meta_r, beta=args.meta_b)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_gamma)
    
    cur_best_pre_0 = 0.
    print('without pretraining.')

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)

    # nonshared_idx = -1

    for epoch in range(args.epoch):
        model.train()

        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]

        t1 = time()
        # loss, rec_loss, emb_loss, ssl_loss, ssl2_loss = 0., 0., 0., 0., 0.
        loss, rec_loss, emb_loss = 0., 0., 0.

        n_batch = int(len(user_train1) / args.batch_size)

        iter_time = time()

        for idx in range(n_batch):
            optimizer.zero_grad()

            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in
                         beh_item_list]  # [[B, max_item1], [B, max_item2], [B, max_item3]]

            u_batch_list, i_batch_list = get_train_pairs(user_train_batch=u_batch,
                                                         beh_item_tgt_batch=beh_batch[-1])  # ndarray[N, ]  ndarray[N, ]

            # load into cuda
            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]

            u_batch_list = torch.from_numpy(u_batch_list).to(device)
            i_batch_list = torch.from_numpy(i_batch_list).to(device)

            model_time = time()

            ua_embeddings, ia_embeddings, rela_embeddings, attn_user, attn_item = model(device)
            batch_rec_loss, batch_emb_loss = recloss(u_batch, beh_batch, ua_embeddings, ia_embeddings, rela_embeddings)
            batch_loss = batch_rec_loss + batch_emb_loss

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item() / n_batch
            rec_loss += batch_rec_loss.item() / n_batch
            emb_loss += batch_emb_loss.item() / n_batch


        # print('iter time: %.1fs' % (time() - iter_time))
        if args.lr_decay: scheduler.step()
        torch.cuda.empty_cache()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:

                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, rec_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        model.eval()
        with torch.no_grad():
            # ua_embeddings, ia_embeddings, _, _, _, _, rela_embeddings, attn_user, attn_item = model(sub_mat, device)
            ua_embeddings, ia_embeddings, rela_embeddings, attn_user, attn_item = model(device)


            users_to_test = list(data_generator.test_set.keys())
            ret = test_torch(ua_embeddings[:, -1, :].detach().cpu().numpy(),
                             ia_embeddings[:, -1, :].detach().cpu().numpy(),
                             rela_embeddings[behs[-1]].detach().cpu().numpy(), users_to_test)
            
            
            
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][0],
                           ret['recall'][1], ret['recall'][2],
                           ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2],
                           ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop, flag = early_stopping_new(ret['recall'][0], cur_best_pre_0,
                                                                              stopping_step, expected_order='acc',
                                                                              flag_step=10)
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
