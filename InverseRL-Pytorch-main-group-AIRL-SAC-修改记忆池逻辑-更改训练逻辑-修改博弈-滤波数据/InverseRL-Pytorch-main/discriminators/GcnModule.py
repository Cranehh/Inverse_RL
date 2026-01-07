import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()

        self.cfg = cfg

        NFR = 256

        NG = cfg.num_graph
        N = cfg.num_boxes
        T = cfg.num_frames

        NFG = 8
        NFG_ONE = NFG
        self.fc_extend = nn.Linear(NFG, NFG)
        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        if cfg.dataset_name == 'volleyball':
            self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([T * N, NFG_ONE]) for i in range(NG)])
        else:
            self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def calc_pairwise_distance_3d(self, X, Y):
        """
        computes pairwise distance between each element
        Args: 
            X: [B,N,D]
            Y: [B,M,D]
        Returns:
            dist: [B,N,M] matrix of euclidean distances
        """
        B = X.shape[0]

        rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
        ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

        dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

        return torch.sqrt(dist)

    def forward(self, graph_boxes_features, boxes_in_flat,type):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        ## NFG为外观特征数量
        B, N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        NFG_ONE = NFG

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 2
        # graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        # graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B, N, 2)  # B*T, N, 2

        graph_boxes_distances = self.calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B, N, N
        if type == 'vehicle':
            position_mask = (graph_boxes_distances > (10))

        else: ## 采取社交距离作为人之间的互动距离
            position_mask = (graph_boxes_distances > (4))

        relation_graph = None
        graph_boxes_features_list = []
        graph_boxes_features = graph_boxes_features.to(torch.float32)
        graph_boxes_features = F.relu(self.fc_extend(graph_boxes_features))
        for i in range(NG):

            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B,N,NFR

            #             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
            #             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)

            relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE
            # one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        # graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list), dim=0)  # B, N, NFG
        graph_boxes_features = torch.mean(torch.stack(graph_boxes_features_list), dim=0)  # B, N, NFG

        return graph_boxes_features, relation_graph



class GCNnet_collective(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(GCNnet_collective, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, 8
        # print(NFG)

        # self.backbone=MyInception_v3(transform_input=False,pretrained=True)

        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad=False

        # self.roi_align=RoIAlign(*self.cfg.crop_size)

        # self.fc_emb_1=nn.Linear(K*K*D,NFB)
        # self.nl_emb_1=nn.LayerNorm([NFB])

        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])

        # self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def get_relation_graph(self, batch_data, type):
        boxes_features_all, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        ##注意这里把数据换乘3维的，第一个数字为batch_size
        B = boxes_features_all.shape[0]
        T = 1
        MAX_N = 6
        NFB = 8
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        ## 换型
        # boxes_features_all = boxes_features_all.reshape(-1, 6, 8)
        # boxes_in = boxes_in.reshape(-1, 6, 2)
        # bboxes_num_in = bboxes_num_in.reshape(-1, )
        # boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        # boxes_in = boxes_in.reshape(B, T, MAX_N, 2)
        # bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,

        actions_scores = []
        activities_scores = []


        for b in range(B):

            N = bboxes_num_in[b]

            boxes_features = boxes_features_all[b]
            boxes_features = torch.tensor(boxes_features.reshape(-1, 6, 8))
            boxes_features = boxes_features.to('cuda')
            boxes_positions = boxes_in[b]
            boxes_positions = torch.tensor(boxes_positions.reshape(-1, 6, 2)).to('cuda')
            boxes_features = boxes_features.reshape(1, T, MAX_N, NFB)
            boxes_positions = boxes_positions.reshape(1, T, MAX_N, 2)
            # bboxes_num_in = bboxes_num_in.reshape(-1, )
            # bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,

            boxes_features = boxes_features[0, :, :N, :].reshape(1, T * N, NFB)

            boxes_positions = boxes_positions[0, :, :N, :].reshape(T * N, 2)

            # boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T * N, NFB)
            #
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 2)


            for i in range(len(self.gcn_list)):
                graph_boxes_features, relation_graph = self.gcn_list[0](boxes_features, boxes_positions, type)
        return relation_graph

    def forward(self, batch_data, type):
        boxes_features_all, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        ##注意这里把数据换乘3维的，第一个数字为batch_size
        B = boxes_features_all.shape[0]
        T = 1
        MAX_N = 6
        NFB = 8
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        ## 换型
        # boxes_features_all = boxes_features_all.reshape(-1, 6, 8)
        # boxes_in = boxes_in.reshape(-1, 6, 2)
        # bboxes_num_in = bboxes_num_in.reshape(-1, )
        # boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        # boxes_in = boxes_in.reshape(B, T, MAX_N, 2)
        # bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,

        actions_scores = []
        activities_scores = []


        for b in range(B):

            N = bboxes_num_in[b]

            boxes_features = boxes_features_all[b]
            boxes_features = torch.tensor(boxes_features.reshape(-1, 6, 8))
            boxes_features = boxes_features.to('cuda')
            boxes_positions = boxes_in[b]
            boxes_positions = torch.tensor(boxes_positions.reshape(-1, 6, 2)).to('cuda')
            boxes_features = boxes_features.reshape(1, T, MAX_N, NFB)
            boxes_positions = boxes_positions.reshape(1, T, MAX_N, 2)
            # bboxes_num_in = bboxes_num_in.reshape(-1, )
            # bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,

            boxes_features = boxes_features[0, :, :N, :].reshape(1, T * N, NFB)

            boxes_positions = boxes_positions[0, :, :N, :].reshape(T * N, 2)

            # boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T * N, NFB)
            #
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 2)


            for i in range(len(self.gcn_list)):
                graph_boxes_features, relation_graph = self.gcn_list[0](boxes_features, boxes_positions, type)
            boxes_features = boxes_features.reshape(1, T * N, NFB)
            boxes_states = graph_boxes_features + boxes_features  # 1, T*N, NFG
            # boxes_states = self.dropout_global(boxes_states)
            boxes_states = boxes_states.to(torch.float32)
            # actn_score = self.fc_actions(boxes_states)
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)

            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)  # T, NFS
            acty_score = self.fc_activities(boxes_states_pooled)  # T, acty_num

            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num

            # actions_scores.append(actn_score)
            activities_scores.append(acty_score)

        # actions_scores = torch.cat(actions_scores, dim=0)
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        if not self.training:
            B = B // 3
            actions_scores = torch.mean(actions_scores.reshape(-1, 3, actions_scores.shape[1]), dim=1)
            activities_scores = torch.mean(activities_scores.reshape(B, 3, -1), dim=1).reshape(B, -1)

        #         print(actions_scores.shape)
        #         print(activities_scores.shape)

        return activities_scores
