import torch
import torch.nn as nn

class PointEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.query_emb = nn.Embedding(100,256)


    #Points , Nx2
    #labels , N,
    #object_ids N,
    # def forward(self, batched_points, batched_labels, batched_object_ids, pos_encoder, label_encoder):
    def forward(self, points_supervision, pos_encoder, label_encoder, no_label_enc, no_pos_enc, device):
        # batch_size = len(batched_points)
        batch_size = len(points_supervision)
        no_pos_enc =False

        embeddings = []
        for idx in range(batch_size):

            #位置编码
            position_embedding = pos_encoder.calc_emb(points_supervision[idx]['point'])

            if no_pos_enc:
                position_embedding = torch.zeros((position_embedding.size())).to(device) 

            query_embedding = position_embedding 
            embeddings.append(query_embedding)


        return embeddings



def build_point_encoder():
    return PointEncoder()
