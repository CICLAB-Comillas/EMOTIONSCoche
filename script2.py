#A PARTIR DE LOS .WAV OBTENER LAS EMOCIONES, BORRAR .WAV Y VISUALIZAR
#ejecutar en visual studio code o similar
import pandas as pd
import numpy as np
import os
import sys
from typing import List
import glob
from pydub import AudioSegment, silence
# import librosa
# import librosa.display
import pandas as pd
import os
import sys
from typing import List
import matplotlib.pyplot as plt
# import librosa
# import librosa.display
import torch
import torch.nn as nn
from IPython.display import Audio
import warnings
import transformers
import shutil
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from datasets import load_dataset, load_metric
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import random
from datasets import load_dataset, load_metric
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor

import librosa
import IPython.display as ipd
import pandas as pd
from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel,
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from transformers import EvalPrediction
from transformers import TrainingArguments
from typing import Any, Dict, Union
from packaging import version
from transformers import (
    Trainer,
    is_apex_available,
)

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast
from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor


def wav_to_emotion(audio_path):

    model_name_or_path = "checkpoint_28_6"

    pooling_mode = "mean"

    label_list = [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fear",
        "disgust",
        "surprise",
    ]
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, "pooling_mode", pooling_mode)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path,
    )
    target_sampling_rate = feature_extractor.sampling_rate

    @dataclass
    class SpeechClassifierOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None

    class HubertClassificationHead(nn.Module):
        """Head for hubert classification task."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class HubertForSpeechClassification(HubertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config
            self.hubert = HubertModel(config)
            self.classifier = HubertClassificationHead(config)
            self.init_weights()

        def freeze_feature_extractor(self):
            self.hubert.feature_extractor._freeze_parameters()

        def merged_strategy(self, hidden_states, mode="mean"):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception(
                    "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
                )

            return outputs

        def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
        ):
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            outputs = self.hubert(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
                The feature_extractor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """

        feature_extractor: Wav2Vec2FeatureExtractor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [
                {"input_values": feature["input_values"]} for feature in features
            ]
            label_features = [feature["labels"] for feature in features]

            d_type = torch.long if isinstance(label_features[0], int) else torch.float

            batch = self.feature_extractor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch["labels"] = torch.tensor(label_features, dtype=d_type)

            return batch

    data_collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor, padding=True
    )

    is_regression = False

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    """Audio Prediction"""
    model = HubertForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio = audio_path

    radess_train2 = []
    audio_array = librosa.load(audio, sr=16000, mono=True)[0]
    mean1 = np.mean(abs(audio_array))
    if mean1==0:
        scores=[0,0,0,0,0,0,0,0]
        return scores
    
    print("Detection-Start---------------------------------------------------------------")

    print(mean1)
    input = feature_extractor(
        raw_speech=audio_array, sampling_rate=16000, padding=True, return_tensors="pt"
    )
    temp = input.input_values.tolist()
    radess_train2.append(temp[0])
    radess_train2 = radess_train2

    prueba_dataset = pd.DataFrame(
        {
            "name": ["nameprueba"],
            "path": ["pathprueba"],
            "emotion": audio,
            "speech": radess_train2,
            "labels": 7,
        }
    )
    prueba_dataset = prueba_dataset.reset_index(drop=True)



    sampling_rate = feature_extractor.sampling_rate

    def predict(speech):
        speech2 = []
        speech2.append(speech)
        input_values = torch.tensor(speech2)
        input_values = input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        max = 0
        maxi = 0
        for i in range(len(scores)):
            if max < scores[i]:
                max = scores[i]
                maxi = i

        print(f"Detected emotion: {config.id2label[maxi]}")
        outputs = [
            {"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"}
            for i, score in enumerate(scores)
        ]
        return [outputs, scores]

    STYLES = """
    <style>
    div.display_data {
        margin: 0 auto;
        max-width: 500px;
    }
    table.xxx {
        margin: 50px !important;
        float: right !important;
        clear: both !important;
    }
    table.xxx td {
        min-width: 300px !important;
        text-align: center !important;
    }
    </style>
    """.strip()

    def prediction(test) -> List[float]:
        setup = {
            "border": 2,
            "show_dimensions": True,
            "justify": "center",
            "classes": "xxx",
            "escape": False,
        }
        print(f"Real emotion: {test['emotion']}")
        speech1 = test["speech"].loc[0]
        speech = np.array(speech1)
        outputs, scores2 = predict(speech1)
        r = pd.DataFrame(outputs)
        return scores2

    scoress = prediction(prueba_dataset)
    scoress = [0.1 if score < 0.1 else score for score in scoress]
    return scoress


#función para obtener la posicion en el coche en función del cuadrante
def obtener_posicion(numero): 
    opciones = { 
        1: "piloto", 
        2: "copiloto", 
        3: "pasajero trasero izda", 
        4: "pasajero trasero dcha" } 
    
    return opciones.get(numero, "Posición inválida")

#función para obtener la gráfica de araña de los 4 cuadrantes
def obtener_grafica4 (scores, grabacion):
    categorias = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    valores1 = [round(scores[0][0] * 100, 3), round(scores[0][1] * 100, 3), round(scores[0][2] * 100, 3), round(scores[0][3] * 100, 3), round(scores[0][4] * 100, 3), round(scores[0][5] * 100, 3), round(scores[0][6] * 100, 3), round(scores[0][7] * 100, 3)]
    valores2 = [round(scores[1][0] * 100, 3), round(scores[1][1] * 100, 3), round(scores[1][2] * 100, 3), round(scores[1][3] * 100, 3), round(scores[1][4] * 100, 3), round(scores[1][5] * 100, 3), round(scores[1][6] * 100, 3), round(scores[1][7] * 100, 3)]
    valores3 = [round(scores[2][0] * 100, 3), round(scores[2][1] * 100, 3), round(scores[2][2] * 100, 3), round(scores[2][3] * 100, 3), round(scores[2][4] * 100, 3), round(scores[2][5] * 100, 3), round(scores[2][6] * 100, 3), round(scores[2][7] * 100, 3)]
    valores4 = [round(scores[3][0] * 100, 3), round(scores[3][1] * 100, 3), round(scores[3][2] * 100, 3), round(scores[3][3] * 100, 3), round(scores[3][4] * 100, 3), round(scores[3][5] * 100, 3), round(scores[3][6] * 100, 3), round(scores[3][7] * 100, 3)]
    
    num_categorias = len(categorias)
    angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
    angulos += angulos[:1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias)
    ax.set_yticklabels([])
    
    ax.set_ylim(0, max(max(valores1), max(valores2), max(valores3), max(valores4)) + 1)
    
    # Repite el primer valor al final para cerrar la figura
    valores1 += valores1[:1]
    valores2 += valores2[:1]
    valores3 += valores3[:1]
    valores4 += valores4[:1]
    
    ax.plot(angulos, valores1, color='blue', label='piloto')
    ax.plot(angulos, valores2, color='red', label= 'copiloto')
    ax.plot(angulos, valores3, color='green', label='pasajero trasero izda')
    ax.plot(angulos, valores4, color='black', label= 'pasajero trasero dcha')

    ax.legend(loc = 'upper left')

    #ax.fill(angulos, valores1, color='skyblue', alpha=0.5)
    #ax.fill(angulos, valores2, color='red', alpha=0.5)
    ax.grid(True)
    
    #plt.show(block=False)
    plt.savefig('fotos/imagen'+grabacion+'.png')

def obtener_graficaPR (scores, grabacion, n):
    pesos = [10,10,10,10,10,10,10,10]
    pesos [n] = 100
    categorias = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    valores1 = [round(scores[0][0] * 100, 3), round(scores[0][1] * 100, 3), round(scores[0][2] * 100, 3), round(scores[0][3] * 100, 3), round(scores[0][4] * 100, 3), round(scores[0][5] * 100, 3), round(scores[0][6] * 100, 3), round(scores[0][7] * 100, 3)]
    valoresR = [round(pesos[0] * 100, 3), round(pesos[1]*100, 3), round(pesos[2] * 100, 3), round(pesos[3]* 100, 3), round(pesos[4] * 100, 3), round(pesos[5] * 100, 3), round(pesos[6] * 100, 3), round(pesos[7] * 100, 3)]
    categorias1 = ['Neutral: '+str(valores1[0])+'%', 'Calm: '+str(valores1[1])+'%', 'Happy: '+str(valores1[2])+'%', 'Sad: '+str(valores1[3])+'%', 'Angry: '+str(valores1[4])+'%', 'Fear: '+str(valores1[5])+'%', 'Disgust: '+str(valores1[6])+'%', 'Surprise: '+str(valores1[7])+'%']
    num_categorias = len(categorias)
    angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
    angulos += angulos[:1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias1)
    ax.set_yticklabels([])
    
    ax.set_ylim(0, max(max(valores1), max(valoresR)) + 1)
    
    # Repite el primer valor al final para cerrar la figura
    valores1 += valores1[:1]
    valoresR += valoresR[:1]
    
    ax.plot(angulos, valores1, color='blue', label='predicción')
    ax.plot(angulos, valoresR, color='red', label= 'realidad')

    ax.legend(loc = 'upper left')

    #ax.fill(angulos, valores1, color='skyblue', alpha=0.5)
    #ax.fill(angulos, valores2, color='red', alpha=0.5)
    ax.grid(True)
    
    #plt.show(block=False)
    plt.savefig('fotos/imagenPR'+grabacion+'.png')

def get_number(emotion):
    emotions_dict = {
        'Neutral': 0,
        'Calm': 1,
        'Happy': 2,
        'Sad': 3,
        'Angry': 4,
        'Fear': 5,
        'Disgust': 6,
        'Surprise': 7
    }
    return emotions_dict.get(emotion, -1)

cont = 1

#bucle while true
while cont<5: 
    # Ruta de los archivos .wav
    carpeta = "Audios/wav/p"+ str(cont)+".*.wav"
    #print(carpeta)

    # Obtener la lista de archivos .wav en la carpeta

    archivos_wav = glob.glob(carpeta)
    #print(len(archivos_wav))

    if len(archivos_wav)==4:
        print("Se encontraron archivos .wav en la carpeta:")
        sc = [[] for _ in range(4)]
        for fila in sc:
            fila.extend([None]*8)
        # Iterar sobre los archivos .wav encontrados
        for archivo in archivos_wav:
            persona = archivo.split("/")[-1].split(".")[1]
            grabacion = archivo.split("/")[-1].split(".")[0][-1]
            scores = wav_to_emotion(archivo)
            #scores1 = [93.452, 49.345,33.4545,63.231,34,34.5643,44.234,32.345]
            #scores2 = [0, 49.345,33.4545,63.231,16.7687,34.5643,10,32.345]
            #scores3 = [93.452, 5,33.4545,63.231,76,34.5643,44.234,32.345]
            #scores4 = [0, 49.345,33.4545,63.231,16.7687,7,44.234,32.345]
            emocion = "Happy"
            n = get_number(emocion)
            if int(persona)==1:
                sc[0]=scores
            elif int(persona)==2:
                sc[1]=scores
            elif int(persona)==3:
                sc[2]=scores
            elif int(persona)==4:
                sc[3]=scores
            # Borrar el archivo .wav
            #///os.remove(archivo)
        print(sc)
        print(grabacion)
        #print(sc)
        obtener_grafica4(sc, grabacion)
        #obtener_graficaPR(sc, grabacion, n)
        carpeta_raw = "Audios/raw/p"+grabacion+".raw"
        #os.remove(carpeta_raw)  

        cont = cont + 1



