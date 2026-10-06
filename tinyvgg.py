from torch.nn import ModuleList
import torch
from torch import nn

"""
Alur Kerja Pytorch dan Arsitektur Convolutional Neural Network (CNN) menggunakan TinyVGG

Sekarang, gambar menggunakan format Tensor (PyTorch)

Tensor : Array multidimensional yang dapat dioperasikan menggunakan tensor core (GPU)

Tujuannya agar gambar bisa diproses super cepat menggunakan GPU

dan kenapa pake CNN? karena gambar yang di flatten akan kehilangan informasi spasial 

(urutan piksel) sehingga tidak efektif untuk klasifikasi gambar.

Flow:
    Input gambar dalam bentuk matriks dimensi tensor [C, H, W]
                            |
                            V
    Proses Convolution (menggunakan Conv2D layer) --> ReLU --> Proses Pooling (menggunakan MaxPool2D layer)
                            |
                            V
             Proses Flatten (meratakan gambar menjadi vector)
                            |
                            V
          Proses Fully Connected (menggunakan Linear layer)
                            |
                            V
            Output gambar dalam bentuk label atau prediksi

Output dari file ini adalah, kita bikin otak penglihatan AI.

Dan AI ini belum bisa melihat, karena belum dilatih, proses selanjutnya adalah melatih AI ini.

"""

# ==============================================================================================================
# 1. Menentukan Device
# ==============================================================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(device)
# print(torch.cuda.get_device_name(0))

# ==============================================================================================================
# 2. Membuat Class Model TinyVGG
# ==============================================================================================================

class TinyVGG(nn.Module):
    """
    Arsitektur TinyVGG ada 2 blok convolusional
    1. Blok Konvolusi pertama: Conv -> ReLU -> Conv -> ReLU -> Maxpool
    2. Blok Konvolusi kedua : Conv -> ReLU -> Conv -> ReLU -> Maxpool

    Diikuti dengan classifier (Flatten -> Linier)
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # Blok konvolusi 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d( # Konvolusi pertama
                in_channels=input_shape, # Channel warna (3 untuk RGB, 1 Untuk Grayscale)
                out_channels=hidden_units,
                kernel_size=3, # Jumlah filter
                stride=1, # Pergerakan 1 pixel per langkah
                padding=1
            ), # Output matriks/bentuk jadi: [jumlah batch, hidden unit, H, W]
            nn.ReLU(), # Biar bikin semua angka minus jadi 0
            nn.Conv2d( # Konvolusi kedua
                in_channels=hidden_units, # Menerima hidden unit dari konvo 1
                out_channels=hidden_units, # Outputnya ngeluarin hidden unit lagi
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(), # Ubah angka minus jadi 0
            nn.MaxPool2d(kernel_size=2, stride=2) # Maxpool2d mengubah H dan W jadi H/2 dan W/2 (dikompres)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # Dikompres lagi jadi H/4 dan W/4
        )

        # Setelah itu buat classifier (Lapisan linier/fully connected)
        # Jikan input gambar 64 x 64, setelah kena maxpool, output akhirnya
        # jadi 16 * 16

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 16 * 16,
                out_features=output_shape
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Masukin gambar ke conv block 1
        x = self.conv_block_1(x)

        # Masukin output dari conv block 1 ke conv block 2
        x = self.conv_block_2(x)

        # Masukin output dari conv block 2 ke classifier
        x = self.classifier(x)

        # Outputnya
        return x

# ==============================================================================================================
# 3. Membuat Instance Model TinyVGG dan Dicekannya ke Device
# ==============================================================================================================

model_0 = TinyVGG(
    input_shape=3, # 3 Channel Warna
    hidden_units=10,
    output_shape=3
).to(device)

# ==============================================================================================================
# 4. Menampilkan Detail Arsitektur Model TinyVGG
# ==============================================================================================================

print("\n==================================================================================\n")
print("Struktur Model\n")
print("==================================================================================")
print(model_0, "\n")

# ==============================================================================================================
# 5. Training Model dengan Dummy Tensor
# ==============================================================================================================

# Setup loss function dan optimizer

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

model_0.train()

# Membuat data torch dummy
dummy_input = torch.randn(size=(32, 3, 64, 64)).to(device)

# 1. Forward pass
y_logits = model_0(dummy_input)

# 2. Hitung loss (asumsi y_target berisi label integer kelas 0,1,2)
y_label = torch.randint(high=3, size=(32,)).to(device)

# Hitung loss menggunakan CrossEntropyLoss
loss = loss_function(y_logits, y_label)

# 3. Reset Gradien
optimizer.zero_grad()

# 4. Backpropagation   
loss.backward() 

# 5. Update weight
optimizer.step() # Mengupdate weight model

print(f"\\nNilai Loss pada iterasi ini: {loss.item():.4f}")

# ==============================================================================================================
# 6. Uji model dengan dummy tensor
# ==============================================================================================================

# Exec prediksi dalam mode inferensi
model_0.eval()
with torch.inference_mode():
    dummy_output = model_0(dummy_input)

print("\\n--- Verifikasi Dimensi Tensor ---")
print(f"Shape Input Dummy : {dummy_input.shape}") # [32, 3, 64, 64]
print(f"Shape Output Model : {dummy_output.shape}") # [32, 3] -> 32 sampel, 3