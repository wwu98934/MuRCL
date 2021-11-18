from torch import nn


class CL(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(CL, self).__init__()
        self.encoder = encoder

        self.projection_dim = projection_dim
        self.n_features = n_features

    def forward(self, x_views):
        assert isinstance(x_views, list), f""
        h_views = [self.encoder(x)[0] for x in x_views]
        return h_views, [h.detach() for h in h_views]
