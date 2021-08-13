import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Perceiver(nn.Module):
    def __init__(
        self,
        num_classes,
        num_hidden_layers=8,
        latent_size=64,
        hidden_size=256,
        num_attention_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        # parameters
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout

        # modules
        self.cross_attention = CrossAttentionLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=self.num_attention_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_hidden_layers
        )
        self.pooling = nn.AvgPool1d(kernel_size=self.latent_size)
        self.classifier = nn.Linear(
            in_features=self.hidden_size, out_features=self.num_classes
        )

        # latent array
        self.latent_array = nn.Parameter(
            torch.rand(self.latent_size, self.hidden_size) * 0.001
        )

    def forward(self, x):

        latent_array = self._init_latent_array(batch_size=x.shape[1])

        out = self.cross_attention(input_array=x, latent_array=latent_array)
        out = self.transformer(out)
        out = self.pooling(out.permute(1, 2, 0)).squeeze()
        out = self.classifier(out).squeeze()
        return out

    def _init_latent_array(self, batch_size):
        return (
            self.latent_array.clone().unsqueeze(1).repeat(1, batch_size, 1).to(DEVICE)
        )


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size=256, num_attention_heads=4, dropout=0.1):
        super().__init__()

        self.latent_layer_norm = nn.LayerNorm(hidden_size)
        self.input_layer_norm = nn.LayerNorm(hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

    def forward(self, input_array, latent_array):

        input_array = self.input_layer_norm(input_array)
        latent_array = self.latent_layer_norm(latent_array)

        q = self.linear_q(latent_array)
        k = self.linear_k(input_array)
        v = self.linear_v(input_array)

        out = self.multi_head_attn(query=q, key=k, value=v)[0]

        out = out + latent_array
        return out


if __name__ == "__main__":

    NUM_CLASSES = 10
    NUM_HIDDEN_LAYERS = 8
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 128
    HIDDEN_SIZE = 256
    LATENT_SIZE = 64
    NUM_ATTENTION_HEADS = 4

    input_tensor = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE)
    print(f"Input tensor shape : {input_tensor.shape}")

    model = Perceiver(
        num_classes=NUM_CLASSES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
    )

    output_tensor = model(input_tensor)

    print(f"Output tensor shape : {output_tensor.shape}")
