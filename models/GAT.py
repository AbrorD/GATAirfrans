# models/GAT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng

class GAT(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GAT, self).__init__()

        self.nb_hidden_layers = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers'] # Ini akan menjadi out_channels per head
        self.bn_bool = hparams['bn_bool']
        self.activation = nn.ReLU()
        self.heads = hparams['heads']
        self.dropout_gat = hparams.get('dropout_gat', 0.6) # Dropout untuk GAT layers

        self.encoder = encoder
        self.decoder = decoder

        # Input GAT layer
        self.in_layer = nng.GATConv(
            in_channels=hparams['encoder'][-1],
            out_channels=self.size_hidden_layers,
            heads=self.heads,
            concat=True, # Output akan menjadi heads * size_hidden_layers
            dropout=self.dropout_gat
        )

        self.hidden_layers = nn.ModuleList()
        for _ in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.GATConv(
                in_channels=self.size_hidden_layers * self.heads, # Karena concat=True dari layer sebelumnya
                out_channels=self.size_hidden_layers,
                heads=self.heads,
                concat=True,
                dropout=self.dropout_gat
            ))

        # Output GAT layer
        # Biasanya, layer GAT terakhir tidak menggunakan concat=True atau menggunakan lebih sedikit head
        # Di sini kita akan menggunakan concat=False dan out_heads=1 (atau jumlah head yang sama dan kemudian dirata-rata)
        # agar outputnya sesuai dengan input decoder.
        self.out_layer = nng.GATConv(
            in_channels=self.size_hidden_layers * self.heads,
            out_channels=hparams['decoder'][0], # Output harus sesuai dengan input decoder
            heads=hparams.get('out_heads', 1), # Bisa dikonfigurasi, default 1
            concat=False, # Agar outputnya adalah decoder[0] bukan decoder[0]*out_heads
            dropout=self.dropout_gat
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            # BN untuk input layer (setelah GATConv)
            self.bn.append(nn.BatchNorm1d(self.size_hidden_layers * self.heads, track_running_stats=False))
            # BN untuk hidden layers
            for _ in range(self.nb_hidden_layers - 1):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers * self.heads, track_running_stats=False))
            # Tidak ada BN setelah out_layer karena outputnya langsung ke decoder.

    def forward(self, data):
        z, edge_index = data.x, data.edge_index
        z = self.encoder(z)
        
        # Input layer
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)
        # z = F.dropout(z, p=self.dropout_gat, training=self.training) # Dropout setelah aktivasi jika GATConv tidak punya

        # Hidden layers
        for i in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[i](z, edge_index)
            if self.bn_bool:
                z = self.bn[i + 1](z)
            z = self.activation(z)
            # z = F.dropout(z, p=self.dropout_gat, training=self.training)

        # Output layer
        z = self.out_layer(z, edge_index)
        # Tidak ada aktivasi atau BN di sini, langsung ke decoder

        z = self.decoder(z)
        return z
        