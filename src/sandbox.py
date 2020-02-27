def create_nodes(df, time_delta=10, stop_delta=5):
    # takes a df and clusters rows based off of specific features and thresholds, these clusters are nodes on our train network graph

    # TODO a route is completely characterized by it's "starting" row,
    ## remove all rows except starting rows and assign node to the rest of the data using RID

    # if node_idx == -1, the row is not yet associated with a node
    df_tmp = df.copy()
    df_tmp["node_idx"] = -1
    df_save = pd.DataFrame(columns=df_tmp.columns)
    node_count = 0

    while len(df_tmp) > 0:
        # init the first row of df_tmp as the current node
        df_tmp.at[0, "node_idx"] = node_count
        node_row = df_tmp[0:1]
        df_save = df_save.append(node_row)
        time_radius = pd.Timedelta(time_delta, units="m")
        node_depart_time = pd.to_datetime(node_row["departure_sched"], format="%H%M")
        node_stops = node_row["stops_in_journey"]

        t1,t2 = node_depart_time - time_radius, node_depart_time + time_radius

        for j in range(len(df_tmp)):
            curr_row = df_tmp[j:j+1]
            # check rows to see if they are associated with the current node
            # same origin destination pair
            if node_row["OD"] == curr_row["OD"]:
                # occur on different days
                if node_row["date"] != curr_row["date"]:
                    # similar scheduled initial departure time
                    curr_depart_time = pd.to_datetime(curr_row["departure_sched"], format="%H%M")
                    if t1 <= curr_depart_time <= t2:
                        # similar number of stops
                        # instead of this we may also choose ranges for the n_stops to be a node
                        # i.e.[0, 5], [5,10], [10, 20], etc.
                        if node_stops-stop_delta <= curr_row["stops_in_journey"] <= node_stops+stop_delta:
                        # associate row j with node
                        df_tmp.at[j, "node_idx"] = node_count
                        df_save.append(curr_row)

        # append same node rows to df_save
        # reduce the size of df_tmp as more rows are assigned to nodes
        df_tmp = df_tmp.loc[df_tmp["node_idx"] != -1]
        node_count += 1


    # TODO reconstruct the full data with node index using RID






































# # class GraphAttentionLayer(nn.Module):
# #     """
# #     https://github.com/Diego999/pyGAT/blob/master/layers.py
# #     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
# #     """

# #     def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
# #         super(GraphAttentionLayer, self).__init__()
# #         self.dropout = dropout
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.alpha = alpha
# #         self.concat = concat

# #         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
# #         nn.init.xavier_uniform_(self.W.data, gain=1.414)
# #         self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
# #         nn.init.xavier_uniform_(self.a.data, gain=1.414)

# #         self.leakyrelu = nn.LeakyReLU(self.alpha)

# #     def forward(self, input, adj):
# #         h = torch.mm(input, self.W)
# #         N = h.size()[0]

# #         hidden = [h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)]

# #         tmp = torch.cat(hidden, dim=1)

# #         a_input = tmp.view(N, -1, 2 * self.out_features)



# #         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

# #         zero_vec = -9e15*torch.ones_like(e)

# #         attention = torch.where(adj > 0, e, zero_vec)

# #         attention = F.softmax(attention, dim=1)

# #         attention = F.dropout(attention, self.dropout, training=self.training)

# #         h_prime = torch.matmul(attention, h)

# #         if self.concat:
# #             return F.elu(h_prime)
# #         else:
# #             return h_prime

# #     def __repr__(self):
# #         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# class SpatialAttentionLayer(nn.Module):
#     def __init(self):
#         super(SpatialAttentionLayer, self).__init__()
#         pass

#     def forward(self, X):
#         pass
#         # X shape = (batch_size, c_in, n_timesteps_in, n_nodes)



