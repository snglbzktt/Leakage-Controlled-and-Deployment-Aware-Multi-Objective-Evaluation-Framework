"""
N-BaIoT TinyML model mimarileri ve model kayıt sistemi.

Kayıtlı modeller:
- tinyml_mlp
- compact_dnn
- tiny_1d_cnn

Bütün modeller:
- [batch_size, input_features] biçiminde float32 girdi kabul eder.
- Softmax uygulanmamış sınıf logitleri üretir.
- CrossEntropyLoss ile doğrudan kullanılabilir.
- Dinamik batch boyutunu destekler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from torch import Tensor, nn


# ==========================================================
# MODEL SPECIFICATION
# ==========================================================

@dataclass(frozen=True)
class ModelSpec:
    """Kayıtlı bir modelin bilimsel ve teknik tanımı."""

    registry_name: str
    display_name: str
    architecture_family: str
    description: str
    order_sensitive: bool
    architecture: dict[str, Any]


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validate_model_dimensions(
    input_features: int,
    num_classes: int,
) -> None:
    """Model boyut parametrelerini doğrular."""

    if input_features <= 0:
        raise ValueError(
            "input_features sıfırdan büyük olmalıdır."
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes en az 2 olmalıdır."
        )


def validate_forward_input(
    inputs: Tensor,
    expected_feature_count: int,
    model_name: str,
) -> None:
    """İleri geçiş giriş tensörünü doğrular."""

    if inputs.ndim != 2:
        raise ValueError(
            f"{model_name} iki boyutlu girdi bekliyor: "
            f"[batch, feature]. Alınan shape={tuple(inputs.shape)}"
        )

    if inputs.shape[1] != expected_feature_count:
        raise ValueError(
            f"{model_name} özellik sayısı uyuşmuyor: "
            f"beklenen={expected_feature_count}, "
            f"alınan={inputs.shape[1]}"
        )


# ==========================================================
# TINYML MLP
# ==========================================================

class TinyMLMLP(nn.Module):
    """
    Küçük tam bağlantılı TinyML sınıflandırıcısı.

    Mimari:
        Input
        Linear(input_features, 64)
        ReLU
        Linear(64, 32)
        ReLU
        Linear(32, num_classes)
    """

    registry_name = "tinyml_mlp"

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_units: tuple[int, int] = (64, 32),
    ) -> None:
        super().__init__()

        validate_model_dimensions(
            input_features=input_features,
            num_classes=num_classes,
        )

        if len(hidden_units) != 2:
            raise ValueError(
                "TinyMLMLP için iki gizli katman boyutu gerekir."
            )

        if any(
            hidden_size <= 0
            for hidden_size in hidden_units
        ):
            raise ValueError(
                "Gizli katman boyutları pozitif olmalıdır."
            )

        first_hidden, second_hidden = hidden_units

        self.input_features = int(
            input_features
        )

        self.num_classes = int(
            num_classes
        )

        self.hidden_units = (
            int(first_hidden),
            int(second_hidden),
        )

        self.network = nn.Sequential(
            nn.Linear(
                self.input_features,
                self.hidden_units[0],
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.Linear(
                self.hidden_units[0],
                self.hidden_units[1],
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.Linear(
                self.hidden_units[1],
                self.num_classes,
            ),
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Sınıf logitlerini üretir."""

        validate_forward_input(
            inputs=inputs,
            expected_feature_count=self.input_features,
            model_name=self.registry_name,
        )

        return self.network(
            inputs
        )


# ==========================================================
# COMPACT DNN
# ==========================================================

class CompactDNN(nn.Module):
    """
    Daha yüksek kapasiteli kompakt tam bağlantılı ağ.

    Mimari:
        Input
        Linear(input_features, 128)
        ReLU
        Dropout(0.10)
        Linear(128, 64)
        ReLU
        Dropout(0.10)
        Linear(64, 32)
        ReLU
        Linear(32, num_classes)

    Dropout yalnızca eğitim sırasında etkindir. Çıkarım sırasında
    parametre, MAC veya model dosyası maliyetini artırmaz.
    """

    registry_name = "compact_dnn"

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_units: tuple[int, int, int] = (
            128,
            64,
            32,
        ),
        dropout_probability: float = 0.10,
    ) -> None:
        super().__init__()

        validate_model_dimensions(
            input_features=input_features,
            num_classes=num_classes,
        )

        if len(hidden_units) != 3:
            raise ValueError(
                "CompactDNN için üç gizli katman boyutu gerekir."
            )

        if any(
            hidden_size <= 0
            for hidden_size in hidden_units
        ):
            raise ValueError(
                "Gizli katman boyutları pozitif olmalıdır."
            )

        if not (
            0.0
            <= dropout_probability
            < 1.0
        ):
            raise ValueError(
                "Dropout olasılığı [0, 1) aralığında olmalıdır."
            )

        self.input_features = int(
            input_features
        )

        self.num_classes = int(
            num_classes
        )

        self.hidden_units = tuple(
            int(hidden_size)
            for hidden_size in hidden_units
        )

        self.dropout_probability = float(
            dropout_probability
        )

        self.network = nn.Sequential(
            nn.Linear(
                self.input_features,
                self.hidden_units[0],
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.Dropout(
                p=self.dropout_probability
            ),
            nn.Linear(
                self.hidden_units[0],
                self.hidden_units[1],
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.Dropout(
                p=self.dropout_probability
            ),
            nn.Linear(
                self.hidden_units[1],
                self.hidden_units[2],
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.Linear(
                self.hidden_units[2],
                self.num_classes,
            ),
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Sınıf logitlerini üretir."""

        validate_forward_input(
            inputs=inputs,
            expected_feature_count=self.input_features,
            model_name=self.registry_name,
        )

        return self.network(
            inputs
        )


# ==========================================================
# TINY 1D CNN
# ==========================================================

class Tiny1DCNN(nn.Module):
    """
    Özellik dizisi üzerinde çalışan küçük 1D-CNN.

    Mimari:
        Input [B, 115]
        Reshape [B, 1, 115]
        Conv1d(1, 8, kernel=5)
        ReLU
        MaxPool1d(2)
        Conv1d(8, 16, kernel=3)
        ReLU
        MaxPool1d(2)
        Conv1d(16, 24, kernel=3)
        ReLU
        AdaptiveAvgPool1d(1)
        Linear(24, num_classes)

    Bu mimari özellik sütunu sırasına duyarlıdır. Özellik sırası bütün
    deneylerde scaler artifactındaki sıra olarak sabitlenmelidir.
    """

    registry_name = "tiny_1d_cnn"

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        channels: tuple[int, int, int] = (
            8,
            16,
            24,
        ),
    ) -> None:
        super().__init__()

        validate_model_dimensions(
            input_features=input_features,
            num_classes=num_classes,
        )

        if input_features < 8:
            raise ValueError(
                "Tiny1DCNN en az 8 özellik gerektirir."
            )

        if len(channels) != 3:
            raise ValueError(
                "Tiny1DCNN için üç kanal boyutu gerekir."
            )

        if any(
            channel_count <= 0
            for channel_count in channels
        ):
            raise ValueError(
                "CNN kanal boyutları pozitif olmalıdır."
            )

        self.input_features = int(
            input_features
        )

        self.num_classes = int(
            num_classes
        )

        self.channels = tuple(
            int(channel_count)
            for channel_count in channels
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.channels[0],
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv1d(
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv1d(
                in_channels=self.channels[1],
                out_channels=self.channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(
                inplace=False
            ),
            nn.AdaptiveAvgPool1d(
                output_size=1
            ),
        )

        self.classifier = nn.Linear(
            self.channels[2],
            self.num_classes,
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Sınıf logitlerini üretir."""

        validate_forward_input(
            inputs=inputs,
            expected_feature_count=self.input_features,
            model_name=self.registry_name,
        )

        channel_first_inputs = (
            inputs.unsqueeze(1)
        )

        extracted_features = (
            self.feature_extractor(
                channel_first_inputs
            )
        )

        flattened_features = (
            extracted_features.flatten(
                start_dim=1
            )
        )

        return self.classifier(
            flattened_features
        )


# ==========================================================
# REGISTRY
# ==========================================================

MODEL_SPECS: dict[str, ModelSpec] = {
    "tinyml_mlp": ModelSpec(
        registry_name="tinyml_mlp",
        display_name="TinyML-MLP",
        architecture_family="multilayer_perceptron",
        description=(
            "Two-hidden-layer low-parameter MLP baseline."
        ),
        order_sensitive=False,
        architecture={
            "hidden_units": [
                64,
                32,
            ],
            "activation": "ReLU",
            "dropout_probability": 0.0,
        },
    ),
    "compact_dnn": ModelSpec(
        registry_name="compact_dnn",
        display_name="Compact-DNN",
        architecture_family="multilayer_perceptron",
        description=(
            "Higher-capacity three-hidden-layer compact DNN."
        ),
        order_sensitive=False,
        architecture={
            "hidden_units": [
                128,
                64,
                32,
            ],
            "activation": "ReLU",
            "dropout_probability": 0.10,
        },
    ),
    "tiny_1d_cnn": ModelSpec(
        registry_name="tiny_1d_cnn",
        display_name="Tiny-1D-CNN",
        architecture_family="one_dimensional_convolution",
        description=(
            "Order-sensitive compact 1D convolutional classifier."
        ),
        order_sensitive=True,
        architecture={
            "channels": [
                8,
                16,
                24,
            ],
            "kernel_sizes": [
                5,
                3,
                3,
            ],
            "pooling": [
                "MaxPool1d(2)",
                "MaxPool1d(2)",
                "AdaptiveAvgPool1d(1)",
            ],
            "activation": "ReLU",
        },
    ),
}


MODEL_FACTORIES: dict[
    str,
    Callable[..., nn.Module],
] = {
    "tinyml_mlp": TinyMLMLP,
    "compact_dnn": CompactDNN,
    "tiny_1d_cnn": Tiny1DCNN,
}


def list_registered_models() -> tuple[str, ...]:
    """Kayıtlı model adlarını sabit sırada döndürür."""

    return tuple(
        MODEL_SPECS.keys()
    )


def get_model_spec(
    model_name: str,
) -> ModelSpec:
    """Model tanımını döndürür."""

    if model_name not in MODEL_SPECS:
        raise KeyError(
            f"Kayıtlı olmayan model: {model_name}. "
            f"Mevcut modeller: {list_registered_models()}"
        )

    return MODEL_SPECS[
        model_name
    ]


def create_model(
    model_name: str,
    input_features: int,
    num_classes: int,
    **model_kwargs: Any,
) -> nn.Module:
    """Kayıt adından yeni model örneği oluşturur."""

    if model_name not in MODEL_FACTORIES:
        raise KeyError(
            f"Kayıtlı olmayan model: {model_name}. "
            f"Mevcut modeller: {list_registered_models()}"
        )

    model_factory = MODEL_FACTORIES[
        model_name
    ]

    model = model_factory(
        input_features=input_features,
        num_classes=num_classes,
        **model_kwargs,
    )

    return model


def get_registry_manifest() -> dict[str, Any]:
    """Model kayıt sistemini JSON uyumlu sözlük olarak döndürür."""

    return {
        model_name: asdict(
            model_spec
        )
        for model_name, model_spec
        in MODEL_SPECS.items()
    }