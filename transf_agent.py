import torch.nn as nn
import torch.nn.functional as F
import torch as th
# from utils.noisy_liner import NoisyLinear
from ..layer.transformer import Transformer
from utils.noisy_liner import NoisyLinear

class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()

        self.args = args
        self.n_agents   = args.n_agents
        # get the number of entities for the agent if specified, otherwise use n_entities
        self.n_entities = getattr(
            self.args,
            "n_entities_obs",
            self.args.n_entities
        )
        self.feat_dim   = args.obs_entity_feats
        self.emb_dim    = args.emb

        # embedder
        self.feat_embedding = nn.Linear(
            self.feat_dim,
            self.emb_dim
        )

        # transformer block
        self.transformer = Transformer(
            args.emb,
            args.heads,
            args.depth,
            args.ff_hidden_mult,
            args.dropout
        )
        if self.args.action_selector == "noisy-new":
            self.q_basic = NoisyLinear(args.emb, args.n_actions, True, device=args.device)
            print("噪声网络")
        else:
            # Working
            # self.fc1 = nn.Linear(input_shape, config["rnn_hidden_dim"])
            # self.fc2 = nn.Linear(config["rnn_hidden_dim"], config["n_actions"])

            # Removing all FFNN between CNN and RNN:
            # self.fc2 = nn.Linear(config["rnn_hidden_dim"], config["n_actions"])
            print("非噪声网络")
            self.q_basic = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):

        # prepare the inputs 
        b, a, _ = inputs.size() # batch_size, agents, features
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)

         # project the embeddings
        embs = self.feat_embedding(inputs)

        # the transformer queries and keys are the input embeddings plus the hidden state
        x = th.cat((hidden_state, embs), 1)

        # get transformer embeddings
        embs = self.transformer.forward(x, x)

        # extract the current hidden state
        h = embs[:, 0:1, :]

        # get the q values
        q = self.q_basic(h)

        return q.view(b, a, -1), h.view(b, a, -1)





