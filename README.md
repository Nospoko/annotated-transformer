## Transformer
Transformer is a deep learning model architecture designed for various natural language
processing tasks.At its core, the Transformer employs self-attention mechanisms to capture
contextual relationships between words, enabling it to process sequences effectively. </br>
It consists of an encoder and a decoder, each composed of stacked layers of self-attention
and feedforward neural networks. The architecture allows for capturing long-range dependencies
without the need for explicit sequential processing.
![image](https://github.com/Nospoko/annotated-transformer/assets/74838859/b622628f-7b3d-4061-ae4d-4fcc7905aba6)

## Training
You can train the model on 1% wmt16 dataset by running
```shell
pip install -r requirements.txt
python train.py
```
You can specialize how many data from wmt16 you want, how to name your project on wandb, what
path to save your model state dict to, on which device to run the training (0 for gpu, 'cpu' for cpu). </br>
For example, you can run the training in colab by cloning the repository and running:
```shell
! pip install -r requirements.txt
! python train.py device=cuda train.batch_size=32 data_slice="25%" run_name="colab-run" file_prefix="wmt16_gpu_model"
```
You have to commit the notebook to download the trained model.

### Important note:
If you wish to change data_slice parameter to be different from the last run, you have to delete
'vocab.pt' file from annotated-transformer directory.

## Examples
You can produce translation examples by running:
```shell
python examples.py n_examples=5 run_name=your_run_name
```
Output for model trained on 1% of the data:
```
Source Text (Input)        : <s> Durch den von Obama <unk> Deal um das iranische <unk> hat sich die Beziehung der beiden weiter verschlechtert . </s>
Target Text (Ground Truth) : <s> The relationship between the two has further deteriorated because of the deal that Obama negotiated on Iran 's atomic programme , . </s>
Model Output               : <s> The <unk> of the <unk> was <unk> , which has been made by the <unk> of the <unk> of the <unk> . </s>
===========
Source Text (Input)        : <s> Das Verhältnis zwischen Obama und Netanyahu ist nicht gerade <unk> . </s>
Target Text (Ground Truth) : <s> The relationship between Obama and Netanyahu is not exactly friendly . </s>
Model Output               : <s> The common policy is not a <unk> and not the <unk> . </s>
===========
Source Text (Input)        : <s> In einem Notruf <unk> Professor <unk> Lamb mit einer etwas <unk> Stimme der Polizei , dass er seine Freundin erschossen habe und dass die Beamten zu seinem Haus kommen <unk> . </s>
Target Text (Ground Truth) : <s> In a 911 call , his voice only slightly shaky , college professor <unk> Lamb told police he had shot his <unk> and officers needed to get over to their house . </s>
Model Output               : <s> In a few words , the <unk> of a <unk> , the <unk> , the <unk> of the <unk> , the <unk> and <unk> , the House has been made . </s>
===========
Source Text (Input)        : <s> In einem Notruf gesteht Professor , seine Freundin erschossen zu haben </s>
Target Text (Ground Truth) : <s> In 911 <unk> , Professor <unk> to Shooting <unk> </s>
Model Output               : <s> There is a number of <unk> <unk> , which has been made in the past . </s>
===========
Source Text (Input)        : <s> Innerhalb des Hauses fanden die Beamten die Leiche von Amy Prentiss und eine <unk> Notiz , die auf einen weißen Block <unk> war : " Mir tut es so leid , ich wollte , ich könnte es rückgängig machen , ich liebte Amy und sie ist die einzige Frau , die mich jemals liebte " . Dies stand nach Angaben der Behörden in dem Brief , und er war von Lamb unterzeichnet . </s>
Target Text (Ground Truth) : <s> <unk> the home , officers found Amy Prentiss ' body and a hand - written note <unk> on a white legal pad : " I am so very sorry I wish I could take it back I loved Amy and she is the only woman who ever loved me , " read the letter authorities say was signed by Lamb . </s>
Model Output               : <s> In this respect , I would like to thank the Commissioner for the <unk> and <unk> , which has been a <unk> , which has been said , I am not very much very much of the fact that the only time has been made , and I am not the only that the Member States ' s ' s ' s only ' s ' s ' s own service</s>
===========
```
