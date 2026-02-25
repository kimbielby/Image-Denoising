from imports import *
from configs import Config

class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with BatchNorm and ReLU.

    Standard building block for U-Net architecture. Each DoubleConv
    block consists of: Conv2D -> BatchNorm2D -> ReLU -> Conv2D ->
    BatchNorm2D -> ReLU.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ) -> None:
        """
        Initialise DoubleConv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DoubleConv block.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Lightweight U-Net for image denoising.

    4-level U-Net architecture with skip connections, suitable for
    training on GPUs with 4-8GB memory. Uses DoubleConv blocks in
    encoder and decoder paths with MaxPool for downsampling and
    ConvTranspose for upsampling.

    Architecture:
        - Encoder: 4 levels with MaxPool for downsampling
        - Bottleneck: Deepest representation at 16x initial features
        - Decoder: 4 levels with skip connections  and ConvTranspose upsampling
        - Output: 1x1 convolution to produce final image

    Attributes:
        encoder1-4 (DoubleConv): Encoder blocks
        pool1-4 (MaxPool2D): Downsampling layers
        bottleneck (DoubleConv): Bottleneck block
        upconv1-4 (ConvTranspose2D): Upsampling layers
        decoder1-4 (DoubleConv): Decoder blocks
        output (Conv2D): Final 1x1 convolution
    """
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            init_features: int = 32
    ) -> None:
        """
        Initialise U-Net model.

        Args:
            in_channels: Number of input channels. Default: 3
            out_channels: Number of output channels. Default: 3
            init_features: Initial feature count. Default: 32
        """
        super().__init__()

        features = init_features

        # Encoder (downsampling path)
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        # Final output layer
        self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.

        Processes input through encoder path, bottleneck and decoder path
        with skip connections to produce denoised output.

        Args:
            x: Input noisy image tensor (B, C, H, W)

        Returns:
            Denoised output image (B, C, H, W)

        Note:
            Input height and width should be divisible by 16 due to 4 pooling layers.
            Output has same spatial dimensions as input.
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return self.output(dec1)

def get_model(device: str | torch.device, config: Config) -> UNet:
    """
    Create and initialise U-Net model.

    Instantiates a U-Net model with default parameters (3 input/output
    channels, 32 initial features), moves it to the specified device and
    prints model statistics.

    Args:
        device: Device to load model on (cuda or cpu)
        config: Configuration object

    Returns:
        UNet: Initialised U-Net model on specified device

    Note:
        Prints model size and parameter count to stdout.
    """
    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        init_features=config.model.init_features
    ).to(device)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**2:.2f} MB (float32)")

    return model

